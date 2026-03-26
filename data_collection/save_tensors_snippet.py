"""
Distillation Data Collection Snippet
=====================================
This snippet should be inserted inside the `simple_test_pts` method of
`mmcv/models/detectors/orion.py` in the Orion codebase, right after the LLM
inference produces `ego_feature` (around line 820 in the original file).

Location in orion.py:
    class Orion:
        def simple_test_pts(self, img_metas, ...):
            ...
            ego_feature = ego_feature.to(torch.float32)
            current_states = ego_feature.unsqueeze(1)

            # <<< INSERT THIS BLOCK HERE >>>

Data saved per frame:
    - visual_queries  : vision encoder output (float16) [1, N_tokens, 4096]
    - planning_token  : LLM planning token / ego_feature (float16) [1, 1, 4096]
    - ego_fut_trajs   : ground-truth future ego trajectories (float16)
    - ego_fut_masks   : future trajectory validity masks (float16)
    - ego_fut_cmd     : driving command (float16)
    - lane_scores     : BEV lane detection scores (float16)
    - lane_preds      : BEV lane predictions, clamped to pc_range (float16)
    - agent_preds     : detected agent bounding box xy positions (float16)
    - agent_fut_preds : agent future trajectory predictions (float16)
    - agent_score_preds     : agent detection confidence scores (float16)
    - agent_fut_cls_preds   : agent future trajectory class scores (float16)

Usage:
    1. Uncomment the block below.
    2. Set `save_dir` to your desired output directory (train or val).
    3. Run Orion inference on your dataset.
    4. Each frame is saved as `data_{scene_token}_{frame_idx}.npz`.

Note:
    The outer `simple_test` entry-point (line ~1086) contains a duplicate
    block with an early-exit guard (`if os.path.exists(filename + ".npz")`).
    Uncomment that block in `simple_test` to skip already-saved frames when
    re-running inference.
"""

# --- [New Code Start] Save Tensors ---
import numpy as np
import os

save_dir = '/mnt/adas7tb/jgu/Orion/distill/distill_data/val'  # change to 'train' for training split
os.makedirs(save_dir, exist_ok=True)

# Unique filename: scene_token + frame_idx
unique_id = img_metas[0].get('frame_idx', 'unknown_ts')
scene_token = img_metas[0].get('scene_token', 'unknown_scene')
safe_scene_token = str(scene_token).replace('/', '_')
filename = f"{save_dir}/data_{safe_scene_token}_{unique_id}"

ego_fut_masks_gt = data.get('ego_fut_masks', None)
ego_fut_cmd_gt   = data.get('ego_fut_cmd', None)
ego_fut_trajs_gt = data.get('ego_fut_trajs', None)

# Extract lane_scores and lane_preds from outs_lane (same as training)
lane_scores_gt = None
lane_preds_gt  = None
if self.with_map_head and 'outs_lane' in locals():
    if 'all_lane_cls_one2one' in outs_lane:
        lane_scores_gt = outs_lane['all_lane_cls_one2one'][-1]
    if 'all_lane_preds_one2one' in outs_lane:
        lane_preds_gt = outs_lane['all_lane_preds_one2one'][-1]
        # Clamp lane predictions to pc_range (same as training preprocessing)
        for p in range(self.map_head.n_control):
            lane_preds_gt[..., 3 * p].clamp_(
                min=self.map_head.pc_range[0], max=self.map_head.pc_range[3])
            lane_preds_gt[..., 3 * p + 1].clamp_(
                min=self.map_head.pc_range[1], max=self.map_head.pc_range[4])
        lane_preds_gt = lane_preds_gt.reshape(
            lane_preds_gt.shape[0], lane_preds_gt.shape[1], -1, 3)[..., :2]

# Extract agent outputs for collision loss (same tensors as training)
agent_outs_gt = None
preds_dicts = outs_bbox
all_cls_scores      = preds_dicts['all_cls_scores']
all_bbox_preds      = preds_dicts['all_bbox_preds']
all_traj_preds      = preds_dicts['all_traj_preds']
all_traj_cls_scores = preds_dicts['all_traj_cls_scores']
batch, num_agent = all_traj_preds[-1].shape[:2]
agent_fut_preds = all_traj_preds[-1].view(
    batch, num_agent,
    self.pts_bbox_head.fut_mode,
    self.pts_bbox_head.fut_ts, 2)
agent_fut_cls_preds = all_traj_cls_scores[-1].view(
    batch, num_agent, self.pts_bbox_head.fut_mode)
agent_outs_gt = {
    'agent_preds':         all_bbox_preds[-1][..., 0:2],
    'agent_fut_preds':     agent_fut_preds,
    'agent_score_preds':   all_cls_scores[-1].sigmoid(),
    'agent_fut_cls_preds': agent_fut_cls_preds.sigmoid(),
}

# Cast to float16 to save disk space
save_data = {
    'visual_queries':  vision_embeded.cpu().numpy().astype(np.float16),
    'planning_token':  current_states.cpu().numpy().astype(np.float16),
}
if ego_fut_trajs_gt is not None:
    save_data['ego_fut_trajs'] = ego_fut_trajs_gt.cpu().numpy().astype(np.float16)
if ego_fut_masks_gt is not None:
    save_data['ego_fut_masks'] = ego_fut_masks_gt.cpu().numpy().astype(np.float16)
if ego_fut_cmd_gt is not None:
    save_data['ego_fut_cmd']   = ego_fut_cmd_gt.cpu().numpy().astype(np.float16)
if lane_scores_gt is not None:
    save_data['lane_scores']   = lane_scores_gt.cpu().numpy().astype(np.float16)
if lane_preds_gt is not None:
    save_data['lane_preds']    = lane_preds_gt.cpu().numpy().astype(np.float16)
if agent_outs_gt is not None:
    if 'agent_preds' in agent_outs_gt:
        save_data['agent_preds'] = agent_outs_gt['agent_preds'].detach().cpu().numpy().astype(np.float16)
    if 'agent_fut_preds' in agent_outs_gt:
        save_data['agent_fut_preds'] = agent_outs_gt['agent_fut_preds'].detach().cpu().numpy().astype(np.float16)
    if 'agent_score_preds' in agent_outs_gt:
        save_data['agent_score_preds'] = agent_outs_gt['agent_score_preds'].detach().cpu().numpy().astype(np.float16)
    if 'agent_fut_cls_preds' in agent_outs_gt:
        save_data['agent_fut_cls_preds'] = agent_outs_gt['agent_fut_cls_preds'].detach().cpu().numpy().astype(np.float16)

np.savez_compressed(filename, **save_data)
# --- [New Code End] ---
