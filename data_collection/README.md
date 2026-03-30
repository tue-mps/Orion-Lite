# Distillation Data Collection

## Overview

Teacher inference data is collected by running the full Orion model on the
training/validation split of the Bench2Drive dataset and saving intermediate
tensors to `.npz` files. Each file corresponds to one driving frame and
contains the vision embeddings, the LLM planning token, and all supervision
signals used by the student training losses.

## File format

Each `.npz` file contains the following keys (all `float16`):

| Key | Shape | Description |
|-----|-------|-------------|
| `visual_queries` | `[1, N_tokens, 4096]` | Vision encoder output (ViT tokens) |
| `planning_token` | `[1, 1, 4096]` | LLM ego-feature / planning token |
| `ego_fut_trajs` | `[1, cmds, fut_ts, 2]` | GT future ego trajectories (x, y) |
| `ego_fut_masks` | `[1, cmds, fut_ts]` | GT trajectory validity masks |
| `ego_fut_cmd` | `[1, 3]` | Driving command (one-hot) |
| `lane_scores` | `[1, N_lanes, 6]` | BEV lane detection scores |
| `lane_preds` | `[1, N_lanes, N_pts, 2]` | BEV lane predictions clamped to `pc_range` |
| `agent_preds` | `[1, N_agents, 2]` | Agent bounding-box xy positions |
| `agent_fut_preds` | `[1, N_agents, fut_mode, fut_ts, 2]` | Agent future trajectories |
| `agent_score_preds` | `[1, N_agents, N_classes]` | Agent detection confidence |
| `agent_fut_cls_preds` | `[1, N_agents, fut_mode]` | Agent future mode probabilities |

Files are named `data_{scene_token}_{frame_idx}.npz` and stored in separate
`train/` and `val/` directories.

## How to collect

Run the collection script from the repo root:

```bash
bash data_collection/collect_distill_data.sh all
```

Useful environment variables:

```bash
ORION_CKPT=/path/to/Orion.pth
DATA_ROOT=/path/to/bench2drive
INFO_ROOT=/path/to/infos
OUT_DIR=/path/to/distill_data
GPU=0
NUM_WORKERS=4
```

## Verify

```python
import numpy as np
d = np.load('distill_data/val/data_<scene>_<frame>.npz')
print(list(d.files))
print(d['visual_queries'].shape)   # (1, N, 4096)
print(d['planning_token'].shape)   # (1, 1, 4096)
```

## Expected dataset size

| Split | Frames | Approx. size |
|-------|--------|--------------|
| train | ~100k | ~150 GB |
| val   | ~10k  | ~15 GB |

Sizes depend on the number of detected agents and lanes per frame.
