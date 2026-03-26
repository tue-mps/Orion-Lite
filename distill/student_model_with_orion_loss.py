"""
OrionMimicModel — lightweight planning decoder that replaces the Orion LLM.

All dependencies are standard PyTorch + local modules (losses.py, vae_utils.py).
No mmcv or Orion installation is required to run this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

try:
    from .losses import (
        build_loss,
        WeightedL1Loss,
        PlanMapBoundLoss,
        PlanCollisionLoss,
        ProbabilisticLoss,
    )
    from .vae_utils import DistributionModule, PredictModel
except ImportError:
    from losses import (
        build_loss,
        WeightedL1Loss,
        PlanMapBoundLoss,
        PlanCollisionLoss,
        ProbabilisticLoss,
    )
    from vae_utils import DistributionModule, PredictModel


class OrionMimicModel(nn.Module):
    """Mimic model that replaces Orion's LLM with a lightweight transformer decoder
    for trajectory planning.

    Architecture:
        1. Input projection:  vision_embeded (B, N, 4096) → (B, N, hidden_dim)
        2. Learnable planning query attended over vision tokens via TransformerDecoder
        3. Output projection: (B, 1, hidden_dim) → ego_feature (B, 4096)
        4. VAE planner: ego_feature → latent distribution → future trajectories

    Args:
        input_dim (int): Dimension of vision tokens (4096 for Orion).
        hidden_dim (int): Internal transformer dimension.
        output_dim (int): Ego-feature dimension (4096 for Orion).
        num_layers (int): Number of TransformerDecoder layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        with_bound_loss (bool): Use lane-boundary planning loss.
        use_col_loss (bool): Use agent-collision planning loss.
        loss_plan_reg (dict): Config for the trajectory regression loss.
        loss_plan_bound (dict): Config for the boundary loss.
        loss_plan_col (dict): Config for the collision loss.
        loss_vae_gen (dict): Config for the VAE KL loss.
        loss_feature_mimic (dict): Config for the feature-mimic loss.
        mimic_loss_type (str): Type of mimic loss — 'l1' | 'l2' | 'kl' | 'huber'.
    """

    def __init__(self,
                 input_dim=4096,
                 hidden_dim=1024,
                 output_dim=4096,
                 num_layers=6,
                 num_heads=16,
                 dropout=0.1,
                 with_bound_loss=True,
                 use_col_loss=False,
                 loss_plan_reg=None,
                 loss_plan_bound=None,
                 loss_plan_col=None,
                 loss_vae_gen=None,
                 loss_feature_mimic=None,
                 mimic_loss_type="l1",
                 layer_dim=4,
                 latent_dim=32,
                 fut_ts=6,
                 ego_fut_mode=6,
                 MIN_LOG_SIGMA=-5.0,
                 MAX_LOG_SIGMA=5.0):
        super().__init__()

        # Defaults for mutable arguments
        if loss_plan_reg    is None: loss_plan_reg    = dict(type="L1Loss",            loss_weight=3.0)
        if loss_plan_bound  is None: loss_plan_bound  = dict(type="PlanMapBoundLoss",  loss_weight=3.0, dis_thresh=1.0)
        if loss_plan_col    is None: loss_plan_col    = dict(type="PlanCollisionLoss", loss_weight=1.0)
        if loss_vae_gen     is None: loss_vae_gen     = dict(type="ProbabilisticLoss", loss_weight=3.0)
        if loss_feature_mimic is None: loss_feature_mimic = dict(type="L1Loss",        loss_weight=1.0)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        self.fut_ts = fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.with_bound_loss = with_bound_loss
        self.use_col_loss = use_col_loss
        self.mimic_loss_type = str(mimic_loss_type).lower()

        # ── Transformer Decoder ──────────────────────────────────────────────
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm_input = nn.LayerNorm(hidden_dim)
        self.planning_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm_output = nn.LayerNorm(output_dim)
        self._init_weights()

        # ── VAE Planner ──────────────────────────────────────────────────────
        self.PROBABILISTIC = True
        self.latent_dim = latent_dim
        self.MIN_LOG_SIGMA = MIN_LOG_SIGMA
        self.MAX_LOG_SIGMA = MAX_LOG_SIGMA
        self.present_distribution_in_channels = output_dim
        self.future_distribution_in_channels = output_dim + 12  # 6 timesteps × 2

        self.present_distribution = DistributionModule(
            self.present_distribution_in_channels, self.latent_dim,
            min_log_sigma=self.MIN_LOG_SIGMA, max_log_sigma=self.MAX_LOG_SIGMA,
        )
        self.future_distribution = DistributionModule(
            self.future_distribution_in_channels, self.latent_dim,
            min_log_sigma=self.MIN_LOG_SIGMA, max_log_sigma=self.MAX_LOG_SIGMA,
        )
        assert self.present_distribution_in_channels % self.layer_dim == 0
        self.predict_model = PredictModel(
            in_channels=self.latent_dim,
            out_channels=self.present_distribution_in_channels,
            hidden_channels=self.present_distribution_in_channels // self.layer_dim,
            num_layers=self.layer_dim,
        )

        ego_fut_decoder = []
        for _ in range(2):
            ego_fut_decoder.append(nn.Linear(8192, 8192))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(nn.Linear(8192, self.ego_fut_mode * 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

        # ── Loss functions ───────────────────────────────────────────────────
        self.loss_plan_reg   = build_loss(loss_plan_reg)
        self.loss_plan_bound = build_loss(loss_plan_bound)
        if self.use_col_loss:
            self.loss_plan_col = build_loss(loss_plan_col)
        self.loss_vae_gen = build_loss(loss_vae_gen)

        if self.mimic_loss_type == "l1":
            self.loss_feature_mimic = build_loss(loss_feature_mimic)
        elif self.mimic_loss_type == "l2":
            self.loss_feature_mimic = nn.MSELoss()
        elif self.mimic_loss_type == "huber":
            self.loss_feature_mimic = nn.HuberLoss(delta=1.0)
        elif self.mimic_loss_type == "kl":
            self.loss_feature_mimic = nn.KLDivLoss(reduction="batchmean")
        else:
            raise ValueError(f"Unsupported mimic_loss_type: {self.mimic_loss_type}")

        self.with_cur = True

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def load_orion_vae_weights(self, ckpt_path, device="cpu"):
        """Load VAE planner weights from an Orion teacher checkpoint.

        Only the VAE components (present_distribution, future_distribution,
        predict_model, ego_fut_decoder) are loaded; the transformer decoder
        is randomly initialised (trained from distillation).
        """
        print(f"Loading Orion VAE weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)

        vae_components = ["present_distribution", "future_distribution",
                          "predict_model", "ego_fut_decoder"]
        my_state = self.state_dict()
        loaded_keys = []
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "")
            for comp in vae_components:
                if clean_k.startswith(comp):
                    if clean_k in my_state and v.shape == my_state[clean_k].shape:
                        my_state[clean_k].copy_(v)
                        loaded_keys.append(clean_k)
                    break
        print(f"Loaded {len(loaded_keys)} VAE keys.")

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, vision_embeded, ego_fut_trajs=None, ego_fut_masks=None,
                ego_fut_cmd=None, lane_preds=None, lane_scores=None,
                agent_outs=None, target_ego_feature=None,
                return_loss=True, return_timing=False):
        batch_size = vision_embeded.size(0)

        # Transformer decoder
        t0 = None
        if return_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        memory = self.norm_input(self.input_proj(vision_embeded))
        tgt = self.planning_query.expand(batch_size, -1, -1)
        out = self.transformer_decoder(tgt, memory)
        predicted_states = self.norm_output(self.output_proj(out))
        ego_feature = predicted_states.squeeze(1)   # (B, 4096)

        decoder_ms = None
        if return_timing and t0 is not None:
            torch.cuda.synchronize()
            decoder_ms = (time.perf_counter() - t0) * 1000.0

        # VAE planner
        t1 = None
        if return_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
            t1 = time.perf_counter()

        current_states = ego_feature.unsqueeze(1)
        distribution_comp = {}

        future_distribution_inputs = None
        if (self.training or return_loss) and ego_fut_trajs is not None:
            future_distribution_inputs = ego_fut_trajs.reshape(batch_size, ego_fut_trajs.shape[1], -1)

        if self.PROBABILISTIC:
            sample, output_distribution = self.distribution_forward(
                current_states, future_distribution_inputs, None,
                force_training_mode=return_loss,
            )
            distribution_comp.update(output_distribution)

        hidden_states = ego_feature.unsqueeze(1)
        states_hs, _ = self.future_states_predict(batch_size, sample, hidden_states, current_states)

        ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
        ego_fut_trajs_list = []
        for i in range(self.fut_ts):
            pred_i = self.ego_fut_decoder(ego_query_hs[i]).reshape(batch_size, self.ego_fut_mode, 2)
            ego_fut_trajs_list.append(pred_i)
        ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2)   # (B, ego_fut_mode, fut_ts, 2)

        outputs = {
            "ego_feature": ego_feature,
            "ego_fut_preds": ego_fut_preds,
            "distribution_comp": distribution_comp,
        }

        if return_timing and t1 is not None:
            torch.cuda.synchronize()
            vae_ms = (time.perf_counter() - t1) * 1000.0
            outputs["latency_breakdown_ms"] = {
                "student_decoder_ms": decoder_ms or 0.0,
                "vae_ms": vae_ms,
                "student_total_ms": (decoder_ms or 0.0) + vae_ms,
            }

        if return_loss:
            if ego_fut_masks is not None:
                while ego_fut_masks.dim() > 2 and ego_fut_masks.shape[1] == 1:
                    ego_fut_masks = ego_fut_masks.squeeze(1)
            if ego_fut_cmd is not None:
                while ego_fut_cmd.dim() > 2 and ego_fut_cmd.shape[1] == 1:
                    ego_fut_cmd = ego_fut_cmd.squeeze(1)
            if ego_fut_trajs is not None and ego_fut_trajs.dim() == 4:
                ego_fut_trajs = ego_fut_trajs.squeeze(1)

            outputs["losses"] = self.compute_losses(
                ego_feature, ego_fut_preds, ego_fut_trajs, ego_fut_masks,
                ego_fut_cmd, lane_preds, lane_scores, agent_outs,
                distribution_comp, target_ego_feature,
            )

        return outputs

    def distribution_forward(self, present_features, future_distribution_inputs=None,
                             noise=None, force_training_mode=False):
        b = present_features.shape[0]
        c = present_features.shape[1]
        present_mu, present_log_sigma = self.present_distribution(present_features)

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            future_features = torch.cat([present_features, future_distribution_inputs], dim=2)
            future_mu, future_log_sigma = self.future_distribution(future_features)

        if noise is None:
            noise = torch.randn_like(present_mu)

        if self.training or force_training_mode:
            mu = future_mu if future_mu is not None else present_mu
            sigma = torch.exp(future_log_sigma if future_log_sigma is not None else present_log_sigma)
        else:
            mu = present_mu
            sigma = torch.exp(present_log_sigma)

        sample = mu + sigma * noise
        sample = sample.permute(0, 2, 1).expand(b, self.latent_dim, c)

        return sample, {
            "present_mu": present_mu,
            "present_log_sigma": present_log_sigma,
            "future_mu": future_mu,
            "future_log_sigma": future_log_sigma,
        }

    def future_states_predict(self, batch_size, sample, hidden_states, current_states):
        future_prediction_input = sample.unsqueeze(0).expand(self.fut_ts, -1, -1, -1)
        future_prediction_input = future_prediction_input.reshape(self.fut_ts, -1, self.latent_dim)

        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_state  = hidden_states.reshape(self.layer_dim, -1, 4096 // self.layer_dim)
        future_states = self.predict_model(future_prediction_input, hidden_state)

        current_states_hs = current_states.unsqueeze(0).repeat(6, 1, 1, 1)
        future_states_hs  = future_states.reshape(self.fut_ts, batch_size, -1, future_states.shape[2])

        if self.with_cur:
            states_hs = torch.cat((current_states_hs, future_states_hs), dim=-1)
        else:
            states_hs = future_states_hs

        return states_hs, future_states_hs

    def compute_losses(self, ego_feature, ego_fut_preds, ego_fut_trajs,
                       ego_fut_masks, ego_fut_cmd, lane_preds, lane_scores,
                       agent_outs, distribution_comp, target_ego_feature):
        losses = {}

        # 1. Feature mimic loss
        if target_ego_feature is not None:
            if self.mimic_loss_type == "kl":
                log_p = F.log_softmax(ego_feature, dim=-1)
                p     = F.softmax(target_ego_feature, dim=-1)
                loss_feature = self.loss_feature_mimic(log_p, p)
            else:
                loss_feature = self.loss_feature_mimic(ego_feature, target_ego_feature)
            losses["loss_feature_mimic"] = torch.nan_to_num(loss_feature)

        # 2-5. Planning losses
        if ego_fut_trajs is not None and ego_fut_masks is not None and ego_fut_cmd is not None:
            ego_fut_gt = ego_fut_trajs.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
            cmd_w  = ego_fut_cmd[..., None, None]
            mask_w = ego_fut_masks[:, None, :, None]
            weight = (cmd_w * mask_w).repeat(1, 1, 1, 2)

            losses["loss_plan_reg"] = torch.nan_to_num(
                self.loss_plan_reg(ego_fut_preds, ego_fut_gt, weight)
            )

            # Select best mode per sample
            B = ego_fut_preds.shape[0]
            if ego_fut_cmd.dim() == 2 and ego_fut_cmd.shape[1] == self.ego_fut_mode:
                cmd = ego_fut_cmd.clone()
                cmd[cmd.sum(dim=1) == 0, 0] = 1.0
                mode_idx = cmd.float().argmax(dim=1)
            else:
                mode_idx = torch.zeros(B, dtype=torch.long, device=ego_fut_preds.device)

            sel_trajs = ego_fut_preds[torch.arange(B, device=ego_fut_preds.device), mode_idx]
            sel_masks = ego_fut_masks

            if self.with_bound_loss and lane_preds is not None and lane_scores is not None:
                losses["loss_plan_bound"] = torch.nan_to_num(
                    self.loss_plan_bound(sel_trajs, lane_preds, lane_scores,
                                         weight=sel_masks, denormalize=False)
                )

            if self.use_col_loss and agent_outs is not None:
                losses["loss_plan_col"] = torch.nan_to_num(
                    self.loss_plan_col(
                        sel_trajs,
                        agent_outs.get("agent_preds"),
                        agent_outs.get("agent_fut_preds"),
                        agent_outs.get("agent_score_preds"),
                        agent_outs.get("agent_fut_cls_preds"),
                        weight=sel_masks[:, :, None].repeat(1, 1, 2),
                    )
                )

            losses["loss_vae_gen"] = torch.nan_to_num(
                self.loss_vae_gen(distribution_comp, ego_fut_masks)
            )

        return losses
