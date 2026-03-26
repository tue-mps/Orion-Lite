"""
Standalone planning loss functions for the Orion student decoder.

Extracted from:
  - mmcv/models/vad_utils/plan_loss.py  (PlanMapBoundLoss, PlanCollisionLoss)
  - mmcv/models/utils/distributions.py  (ProbabilisticLoss)
  - mmcv/models/losses/utils.py         (weighted_loss, weight_reduce_loss)

All mmcv dependencies have been removed — only standard PyTorch is required.
"""

import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss reduction helpers (replaces mmcv.models.losses.utils)
# ---------------------------------------------------------------------------

def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
    else:
        if reduction == "mean":
            return loss.sum() / avg_factor
        elif reduction == "sum":
            return loss.sum()
    return loss


def weighted_loss(loss_func):
    """Decorator: add weight / reduction / avg_factor args to an element-wise loss."""
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    return wrapper


# ---------------------------------------------------------------------------
# Weighted L1 Loss (replaces mmcv L1Loss with weight support)
# ---------------------------------------------------------------------------

class WeightedL1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        reduction = reduction_override or "mean"
        loss = F.l1_loss(pred, target, reduction="none")
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return self.loss_weight * loss


# ---------------------------------------------------------------------------
# Planning losses
# ---------------------------------------------------------------------------

def segments_intersect(line1_start, line1_end, line2_start, line2_end):
    dx1 = line1_end[:, 0] - line1_start[:, 0]
    dy1 = line1_end[:, 1] - line1_start[:, 1]
    dx2 = line2_end[:, 0] - line2_start[:, 0]
    dy2 = line2_end[:, 1] - line2_start[:, 1]

    det = dx1 * dy2 - dx2 * dy1
    parallel_mask = det == 0

    t1 = ((line2_start[:, 0] - line1_start[:, 0]) * dy2
          - (line2_start[:, 1] - line1_start[:, 1]) * dx2) / det.clamp(min=1e-8)
    t2 = ((line2_start[:, 0] - line1_start[:, 0]) * dy1
          - (line2_start[:, 1] - line1_start[:, 1]) * dx1) / det.clamp(min=1e-8)

    intersect_mask = (t1 >= 0) & (t1 <= 1) & (t2 >= 0) & (t2 <= 1)
    intersect_mask[parallel_mask] = False
    return intersect_mask


@weighted_loss
def plan_map_bound_loss(pred, target, dis_thresh=1.0):
    """Element-wise planning map-boundary constraint loss.

    Args:
        pred   (Tensor): ego_fut_preds [B, fut_ts, 2]
        target (Tensor): lane_bound_preds [B, num_vec, num_pts, 2]

    Returns:
        Tensor: [B, fut_ts]
    """
    pred = pred.cumsum(dim=-2)
    ego_traj_starts = pred[:, :-1, :]
    ego_traj_ends = pred
    B, T, _ = ego_traj_ends.size()
    padding_zeros = torch.zeros((B, 1, 2), dtype=pred.dtype, device=pred.device)
    ego_traj_starts = torch.cat((padding_zeros, ego_traj_starts), dim=1)
    _, V, P, _ = target.size()

    ego_traj_expanded = ego_traj_ends.unsqueeze(2).unsqueeze(3)
    maps_expanded = target.unsqueeze(1)
    dist = torch.linalg.norm(ego_traj_expanded - maps_expanded, dim=-1)
    dist = dist.min(dim=-1, keepdim=False)[0]
    min_inst_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for _ in range(dist.shape[0])]

    bd_target = target.unsqueeze(1).repeat(1, pred.shape[1], 1, 1, 1)
    min_bd_insts = bd_target[batch_idxs, ts_idxs, min_inst_idxs]
    bd_inst_starts = min_bd_insts[:, :, :-1, :].flatten(0, 2)
    bd_inst_ends = min_bd_insts[:, :, 1:, :].flatten(0, 2)
    ego_traj_starts_r = ego_traj_starts.unsqueeze(2).repeat(1, 1, P - 1, 1).flatten(0, 2)
    ego_traj_ends_r = ego_traj_ends.unsqueeze(2).repeat(1, 1, P - 1, 1).flatten(0, 2)

    intersect_mask = segments_intersect(ego_traj_starts_r, ego_traj_ends_r, bd_inst_starts, bd_inst_ends)
    intersect_mask = intersect_mask.reshape(B, T, P - 1).any(dim=-1)
    intersect_idx = (intersect_mask == True).nonzero()

    target_flat = target.view(target.shape[0], -1, target.shape[-1])
    dist2 = torch.linalg.norm(pred[:, :, None, :] - target_flat[:, None, :, :], dim=-1)
    min_idxs = torch.argmin(dist2, dim=-1).tolist()
    min_dist = dist2[batch_idxs, ts_idxs, min_idxs]
    loss = min_dist
    loss[loss > dis_thresh] = 0
    loss[loss <= dis_thresh] = dis_thresh - loss[loss <= dis_thresh]

    for idx in intersect_idx:
        loss[idx[0], idx[1]:] = 0

    return loss


class PlanMapBoundLoss(nn.Module):
    """Planning boundary constraint: penalises ego trajectory that is too close
    to lane boundaries.

    Args:
        loss_weight (float): Loss multiplier.
        dis_thresh (float): Minimum safe distance to lane boundary (metres).
        map_thresh (float): Lane-boundary confidence threshold.
        lane_bound_cls_idx (int): Class index for lane boundaries in lane_scores.
        perception_detach (bool): Whether to detach lane predictions.
    """

    def __init__(self, reduction="mean", loss_weight=1.0, map_thresh=0.5,
                 lane_bound_cls_idx=2, dis_thresh=1.0,
                 point_cloud_range=None, perception_detach=False):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.map_thresh = map_thresh
        self.lane_bound_cls_idx = lane_bound_cls_idx
        self.dis_thresh = dis_thresh
        self.pc_range = point_cloud_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.perception_detach = perception_detach

    def forward(self, ego_fut_preds, lane_preds, lane_score_preds,
                weight=None, avg_factor=None, reduction_override=None, denormalize=True):
        reduction = reduction_override or self.reduction
        if self.perception_detach:
            lane_preds = lane_preds.detach()
            lane_score_preds = lane_score_preds.detach()

        not_lane_bound_mask = lane_score_preds[..., self.lane_bound_cls_idx] < self.map_thresh
        lane_bound_preds = lane_preds.clone()
        if denormalize:
            lane_bound_preds[..., 0:1] = (lane_bound_preds[..., 0:1]
                                          * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            lane_bound_preds[..., 1:2] = (lane_bound_preds[..., 1:2]
                                          * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        lane_bound_preds[not_lane_bound_mask] = 1e6

        return self.loss_weight * plan_map_bound_loss(
            ego_fut_preds, lane_bound_preds,
            weight=weight, dis_thresh=self.dis_thresh,
            reduction=reduction, avg_factor=avg_factor,
        )


@weighted_loss
def plan_col_loss(pred, target, agent_fut_preds, x_dis_thresh=1.5, y_dis_thresh=3.0, dis_thresh=3.0):
    """Element-wise ego-agent collision constraint loss.

    Args:
        pred            (Tensor): ego_fut_preds [B, fut_ts, 2]
        target          (Tensor): agent_preds   [B, num_agent, 2]
        agent_fut_preds (Tensor): [B, num_agent, fut_ts, 2]

    Returns:
        Tensor: [B, fut_ts, 2]
    """
    pred = pred.cumsum(dim=-2)
    agent_fut_preds = agent_fut_preds.cumsum(dim=-2)
    target = target[:, :, None, :] + agent_fut_preds

    dist = torch.linalg.norm(pred[:, None, :, :] - target, dim=-1)
    target[dist > dis_thresh] = 1e6

    x_dist = torch.abs(pred[:, None, :, 0] - target[..., 0])
    y_dist = torch.abs(pred[:, None, :, 1] - target[..., 1])
    x_min_idxs = torch.argmin(x_dist, dim=1).tolist()
    y_min_idxs = torch.argmin(y_dist, dim=1).tolist()
    B = y_dist.shape[0]
    batch_idxs = [[i] for i in range(B)]
    ts_idxs = [[i for i in range(y_dist.shape[-1])] for _ in range(B)]

    x_min_dist = x_dist[batch_idxs, x_min_idxs, ts_idxs]
    y_min_dist = y_dist[batch_idxs, y_min_idxs, ts_idxs]

    x_loss = x_min_dist.clone()
    x_loss[x_loss > x_dis_thresh] = 0
    x_loss[x_loss <= x_dis_thresh] = x_dis_thresh - x_loss[x_loss <= x_dis_thresh]

    y_loss = y_min_dist.clone()
    y_loss[y_loss > y_dis_thresh] = 0
    y_loss[y_loss <= y_dis_thresh] = y_dis_thresh - y_loss[y_loss <= y_dis_thresh]

    return torch.cat([x_loss.unsqueeze(-1), y_loss.unsqueeze(-1)], dim=-1)


class PlanCollisionLoss(nn.Module):
    """Planning collision constraint: penalises ego trajectory that is too close
    to other detected agents.

    Args:
        loss_weight (float): Loss multiplier.
        agent_thresh (float): Agent detection confidence threshold.
        x_dis_thresh (float): Safe lateral distance (metres).
        y_dis_thresh (float): Safe longitudinal distance (metres).
    """

    def __init__(self, reduction="mean", loss_weight=1.0, agent_thresh=0.5,
                 x_dis_thresh=1.5, y_dis_thresh=3.0, point_cloud_range=None):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.agent_thresh = agent_thresh
        self.x_dis_thresh = x_dis_thresh
        self.y_dis_thresh = y_dis_thresh
        self.pc_range = point_cloud_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    def forward(self, ego_fut_preds, agent_preds, agent_fut_preds,
                agent_score_preds, agent_fut_cls_preds,
                weight=None, avg_factor=None, reduction_override=None):
        reduction = reduction_override or self.reduction

        agent_max_score, agent_max_idx = agent_score_preds.max(dim=-1)
        agent_fut_preds = agent_fut_preds.clone()
        agent_fut_preds[agent_max_score < self.agent_thresh] = 1e6
        agent_fut_preds[agent_max_idx > 4] = 1e6  # keep only vehicle classes 0-4

        best_mode_idxs = torch.argmax(agent_fut_cls_preds, dim=-1).tolist()
        B = agent_fut_cls_preds.shape[0]
        batch_idxs = [[i] for i in range(B)]
        agent_idxs = [[j for j in range(agent_fut_cls_preds.shape[1])] for _ in range(B)]
        agent_fut_preds = agent_fut_preds[batch_idxs, agent_idxs, best_mode_idxs]

        return self.loss_weight * plan_col_loss(
            ego_fut_preds, agent_preds,
            agent_fut_preds=agent_fut_preds,
            weight=weight,
            x_dis_thresh=self.x_dis_thresh,
            y_dis_thresh=self.y_dis_thresh,
            reduction=reduction,
            avg_factor=avg_factor,
        )


class ProbabilisticLoss(nn.Module):
    """KL divergence loss between present and future VAE distributions.

    Args:
        loss_weight (float): Loss multiplier.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, output, valid_mask):
        """
        Args:
            output (dict): must contain 'present_mu', 'present_log_sigma',
                           'future_mu', 'future_log_sigma'.
            valid_mask (Tensor): [B, fut_ts]  — frames with valid GT trajectories.
        """
        present_mu        = output["present_mu"]
        present_log_sigma = output["present_log_sigma"]
        future_mu         = output["future_mu"]
        future_log_sigma  = output["future_log_sigma"]

        var_future  = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
            present_log_sigma - future_log_sigma - 0.5
            + (var_future + (future_mu - present_mu) ** 2) / (2 * var_present)
        )
        kl_div = kl_div * valid_mask.any(dim=-1).unsqueeze(-1).unsqueeze(-1)
        return torch.mean(torch.sum(kl_div, dim=-1)) * self.loss_weight


# ---------------------------------------------------------------------------
# Simple build_loss factory (replaces mmcv.models.builder.build_loss)
# ---------------------------------------------------------------------------

def build_loss(cfg: dict) -> nn.Module:
    """Create a loss module from a config dict.

    Supported types:
        'L1Loss'           → WeightedL1Loss
        'PlanMapBoundLoss' → PlanMapBoundLoss
        'PlanCollisionLoss'→ PlanCollisionLoss
        'ProbabilisticLoss'→ ProbabilisticLoss
    """
    cfg = dict(cfg)  # copy
    loss_type = cfg.pop("type")

    registry = {
        "L1Loss":            WeightedL1Loss,
        "PlanMapBoundLoss":  PlanMapBoundLoss,
        "PlanCollisionLoss": PlanCollisionLoss,
        "ProbabilisticLoss": ProbabilisticLoss,
    }

    if loss_type not in registry:
        raise ValueError(f"Unknown loss type '{loss_type}'. "
                         f"Available: {list(registry)}")

    return registry[loss_type](**cfg)
