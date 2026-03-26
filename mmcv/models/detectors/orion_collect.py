"""
OrionCollect — distillation data collection detector.

Drop-in replacement for Orion that saves teacher tensors to .npz files
during inference.  No modifications to orion.py are needed.

Key tensors saved per frame:
    visual_queries     (1, N_tokens, 4096)  vision encoder output
    planning_token     (1, 1, 4096)         LLM ego-feature
    ego_fut_trajs      (1, cmds, fut_ts, 2) GT future ego trajectories
    ego_fut_masks      (1, cmds, fut_ts)    GT trajectory validity masks
    ego_fut_cmd        (1, 3)               Driving command (one-hot)
    lane_scores        (1, N_lanes, 6)      BEV lane detection scores
    lane_preds         (1, N_lanes, N_pts, 2) BEV lane predictions
    agent_preds        (1, N_agents, 2)     Agent xy positions
    agent_fut_preds    (1, N_agents, M, T, 2) Agent future trajectories
    agent_score_preds  (1, N_agents, C)     Agent detection confidence
    agent_fut_cls_preds(1, N_agents, M)     Agent future mode probs

Usage — config snippet:
    model = dict(
        type='OrionCollect',
        save_dir='/path/to/distill_data/val',
        # ... all other Orion kwargs unchanged ...
    )

CLI runner:
    python data_collection/run_orion_collect.py \\
        --config  adzoo/orion/configs/orion_stage3_infer.py \\
        --checkpoint ckpts/Orion.pth \\
        --save-dir  distill_data/val \\
        --split     val \\
        --gpu       0
"""

import os
import numpy as np
import torch

from mmcv.models import DETECTORS
from mmcv.models.detectors.orion import Orion


# ---------------------------------------------------------------------------
# OrionCollect detector
# ---------------------------------------------------------------------------

@DETECTORS.register_module()
class OrionCollect(Orion):
    """Orion with teacher-tensor saving built in.

    Registers forward hooks on the map head, detection head and LLM to
    capture intermediate tensors, then saves them as .npz files after
    each frame's inference.

    Args:
        save_dir (str): Directory to write .npz files into.
        **kwargs: Passed directly to Orion.__init__.
    """

    def __init__(self, save_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._cap = {}          # tensor capture buffer
        self._hooks = []        # registered hook handles (for cleanup)
        self._install_hooks()

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _install_hooks(self):
        """Register forward hooks to capture intermediate tensors."""

        def _primary_output(output):
            if isinstance(output, (tuple, list)) and len(output) > 0:
                return output[0]
            return output

        # Map head → outs_lane dict
        if self.with_map_head:
            def _map_hook(module, input, output):
                self._cap['outs_lane'] = _primary_output(output)
            self._hooks.append(
                self.map_head.register_forward_hook(_map_hook)
            )

        # Detection head → outs_bbox dict
        def _bbox_hook(module, input, output):
            self._cap['outs_bbox'] = _primary_output(output)
        self._hooks.append(
            self.pts_bbox_head.register_forward_hook(_bbox_hook)
        )

        # LLM inference_ego → vision_embeded (images arg) + ego_feature (return)
        original_inference_ego = self.lm_head.inference_ego

        def _patched_inference_ego(inputs, images=None, **kw):
            result = original_inference_ego(inputs=inputs, images=images, **kw)
            self._cap['vision_embeded'] = images
            self._cap['ego_feature'] = result
            return result

        self.lm_head.inference_ego = _patched_inference_ego

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Override simple_test to add early-exit + save after inference
    # ------------------------------------------------------------------

    def simple_test(self, img_metas, **data):
        """Run inference, save tensors, return normal results."""
        self._cap.clear()

        # --- early-exit: skip already-saved frames ---
        unique_id   = img_metas[0].get('frame_idx', 'unknown_ts')
        scene_token = img_metas[0].get('scene_token', 'unknown_scene')
        safe_token  = str(scene_token).replace('/', '_')
        filename    = os.path.join(self.save_dir, f"data_{safe_token}_{unique_id}")

        if os.path.exists(filename + ".npz"):
            print(f"[OrionCollect] skip (exists): {filename}.npz")
            return [dict() for _ in img_metas]

        # --- run full Orion inference (hooks fire during this call) ---
        results = super().simple_test(img_metas, **data)

        # --- save tensors if hooks captured what we need ---
        if 'vision_embeded' not in self._cap or 'ego_feature' not in self._cap:
            return results

        vision_embeded = self._cap['vision_embeded']
        ego_feature    = self._cap['ego_feature']
        current_states = ego_feature.unsqueeze(1) if ego_feature.dim() == 2 else ego_feature

        save_data = {
            'visual_queries': vision_embeded.detach().cpu().numpy().astype(np.float16),
            'planning_token': current_states.detach().cpu().numpy().astype(np.float16),
        }

        # GT trajectory data
        for key in ('ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd'):
            val = data.get(key, None)
            if val is not None and isinstance(val, torch.Tensor):
                save_data[key] = val.detach().cpu().numpy().astype(np.float16)

        # Lane predictions
        outs_lane = self._cap.get('outs_lane', None)
        if outs_lane is not None:
            lane_scores = outs_lane.get('all_lane_cls_one2one', None)
            lane_preds  = outs_lane.get('all_lane_preds_one2one', None)
            if lane_scores is not None:
                save_data['lane_scores'] = lane_scores[-1].detach().cpu().numpy().astype(np.float16)
            if lane_preds is not None:
                lp = lane_preds[-1]
                for p in range(self.map_head.n_control):
                    lp[..., 3 * p    ].clamp_(min=self.map_head.pc_range[0], max=self.map_head.pc_range[3])
                    lp[..., 3 * p + 1].clamp_(min=self.map_head.pc_range[1], max=self.map_head.pc_range[4])
                lp = lp.reshape(lp.shape[0], lp.shape[1], -1, 3)[..., :2]
                save_data['lane_preds'] = lp.detach().cpu().numpy().astype(np.float16)

        # Agent predictions
        outs_bbox = self._cap.get('outs_bbox', None)
        if outs_bbox is not None:
            all_cls_scores    = outs_bbox.get('all_cls_scores', None)
            all_bbox_preds    = outs_bbox.get('all_bbox_preds', None)
            all_traj_preds    = outs_bbox.get('all_traj_preds', None)
            all_traj_cls      = outs_bbox.get('all_traj_cls_scores', None)
            if all_bbox_preds is not None:
                save_data['agent_preds'] = (
                    all_bbox_preds[-1][..., 0:2].detach().cpu().numpy().astype(np.float16)
                )
            if all_traj_preds is not None:
                B, N = all_traj_preds[-1].shape[:2]
                fut_mode = self.pts_bbox_head.fut_mode
                fut_ts   = self.pts_bbox_head.fut_ts
                agent_fut = all_traj_preds[-1].view(B, N, fut_mode, fut_ts, 2)
                save_data['agent_fut_preds'] = agent_fut.detach().cpu().numpy().astype(np.float16)
            if all_cls_scores is not None:
                save_data['agent_score_preds'] = (
                    all_cls_scores[-1].sigmoid().detach().cpu().numpy().astype(np.float16)
                )
            if all_traj_cls is not None:
                B, N = all_traj_cls[-1].shape[:2]
                fut_mode = self.pts_bbox_head.fut_mode
                agent_fut_cls = all_traj_cls[-1].view(B, N, fut_mode)
                save_data['agent_fut_cls_preds'] = (
                    agent_fut_cls.sigmoid().detach().cpu().numpy().astype(np.float16)
                )

        np.savez_compressed(filename, **save_data)
        print(f"[OrionCollect] saved: {filename}.npz  keys={list(save_data)}")
        return results
