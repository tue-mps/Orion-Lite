# Distillation Data Collection

## Overview

Teacher inference data is collected by running the full Orion model on the
training/validation split of the Bench2Drive dataset and saving intermediate
tensors to `.npz` files.  Each file corresponds to one driving frame and
contains the vision embeddings, the LLM planning token, and all ground-truth
supervision signals used by the student training losses.

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

### Step 1 – Patch `orion.py`

Open `mmcv/models/detectors/orion.py` in this repo (it is at `mmcv/models/detectors/orion.py`)
and locate the `simple_test_pts` method.  Find the line:

```python
current_states = ego_feature.unsqueeze(1)
```

Immediately after this line, uncomment (or insert) the save-tensors block from
[`save_tensors_snippet.py`](save_tensors_snippet.py).  Set `save_dir` to your
desired output path:

```python
save_dir = '/path/to/distill_data/train'   # for the training split
# save_dir = '/path/to/distill_data/val'   # for the validation split
```

Additionally, uncomment the duplicate block near the `simple_test` entry-point
(~line 1086) which adds an early-exit guard to skip already-saved frames:

```python
if os.path.exists(filename + ".npz"):
    print(f"file already exist {filename}")
    return [dict() for i in range(len(img_metas))]
```

### Step 2 – Run inference

Use the standard Orion evaluation pipeline.  Run from the repo root:

```bash
# collect training split
python adzoo/orion/test.py \
    adzoo/orion/configs/orion_b2d.py \
    ckpts/Orion.pth \
    --eval bbox \
    --data-split train

# collect validation split
python adzoo/orion/test.py \
    adzoo/orion/configs/orion_b2d.py \
    ckpts/Orion.pth \
    --eval bbox \
    --data-split val
```

### Step 3 – Verify

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
