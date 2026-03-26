# Orion-Lite

Knowledge distillation of the [Orion](https://github.com/...) autonomous driving model into a lightweight planning decoder.  The full LLM-based planning module is replaced by a small transformer decoder trained by distillation from teacher Orion inference data.

## Overview

```
Orion (teacher)
  ├── EVA-ViT vision backbone   ──────────────────────┐
  ├── LLM (planning token generation)                 │  frozen at eval
  ├── BEV map head                                    │
  └── Detection head                                  │
                                                      ▼
Orion-Lite (student)
  └── Lightweight transformer decoder  ←── distilled from teacher
        • configurable depth: 2 / 4 / 6 / 8 / 16 layers
        • optional feature-mimic loss during training
```

The student decoder takes the vision encoder output as input and directly
predicts the ego planning feature, bypassing the expensive LLM forward pass at
inference time.

---

## Repository structure

```
Orion-Lite/
├── mmcv/              # Orion's custom mmcv fork (vision backbone, model heads, losses)
├── adzoo/             # Orion training/testing entry-points and configs
│   └── orion/
│       ├── train.py, test.py, apis/
│       └── configs/   (base Orion configs)
├── team_code/         # Bench2Drive driving agent (orion_b2d_agent_strict.py)
├── setup.py           # Build mmcv C extensions: pip install -v -e . --no-deps
├── requirements.txt   # All dependencies
│
├── data_collection/
│   ├── README.md                   # How to collect distillation data
│   └── save_tensors_snippet.py     # Patch for mmcv/models/detectors/orion.py
│
├── distill/
│   ├── train_student.py                    # Basic distillation (MSE/KL/L1/Huber loss)
│   ├── train_student_with_orion_loss.py    # Full distillation with planning losses
│   ├── train_ablation_decoder_layers.py    # Multi-GPU ablation: 2/4/8/16 layers
│   ├── student_model.py                    # OrionStudent: pure transformer decoder
│   ├── student_model_with_orion_loss.py    # OrionMimicModel: decoder + VAE + planning losses
│   ├── losses.py                           # Standalone planning losses (no mmcv dependency)
│   ├── vae_utils.py                        # Standalone VAE modules (no mmcv dependency)
│   ├── dataloader.py                       # NpzDistillDataset
│   │
│   └── results/
│       ├── orion_student_no_mimic_loss/    # 6-layer decoder, no mimic loss
│       ├── orion_student_with_mimic_loss/  # 6-layer decoder, with mimic (L1) loss
│       ├── ablation_decoder_layers_no_mimic/
│       │   ├── decoder2l/   decoder4l/   decoder8l/   decoder16l/
│       └── ablation_decoder_layers_with_mimic/
│           ├── decoder2l/   decoder4l/   decoder8l/   decoder16l/
│
└── eval/
    ├── configs/
    │   ├── base/                          # Full OrionStudentPlanner base configs
    │   ├── orion_student_exp{7,8,9}_stage2_b2d_agent_eval.py
    │   ├── orion_student_6l_{no,with}_mimic_b2d_agent_eval.py
    │   └── orion_student_decoder{2,4,8,16}l_{no,with}_mimic_b2d_agent_eval.py
    │
    └── scripts/
        ├── run_orion_student_exp9_multi_strict.sh
        ├── run_orion_student_exp7_exp8_multi_strict.sh
        ├── run_orion_student_decoder_ablation_multi_strict.sh
        ├── run_orion_student_no_mimic_loss_multi_strict.sh
        ├── run_orion_student_with_mimic_loss_multi_strict.sh
        └── export_distill_ckpt.py         # Merge backbone + distill decoder → full ckpt
```

---

## Step 1 – Collect distillation data

See [data_collection/README.md](data_collection/README.md) for a detailed guide.

In short: patch [`mmcv/models/detectors/orion.py`](mmcv/models/detectors/orion.py)
with the snippet in [`data_collection/save_tensors_snippet.py`](data_collection/save_tensors_snippet.py),
then run Orion inference from this repo.  Each frame is saved as a compressed `.npz`
file containing the vision tokens, the teacher planning token, and all
ground-truth supervision signals.

---

## Step 2 – Train the student decoder

### Basic training (MSE/L1/Huber/KL distillation loss only)

```bash
cd distill
python train_student.py \
    --train_dir distill_data/train \
    --val_dir   distill_data/val \
    --loss      mse \
    --epochs    20 \
    --hidden_dim 1024 \
    --num_layers 6
```

### Full training with Orion planning losses

Trains `OrionMimicModel` with planning regulation, boundary, collision, and
optional feature-mimic losses.

```bash
cd distill
python train_student_with_orion_loss.py \
    --train-dir distill_data/train \
    --val-dir   distill_data/val \
    --orion-ckpt /path/to/Orion.pth \   # optional: initialise VAE from Orion ckpt
    --use-feature-mimic-loss \
    --epochs 20 \
    --run-name orion_student_with_mimic_loss
```

Drop `--use-feature-mimic-loss` to train without the feature-mimic loss:

```bash
python train_student_with_orion_loss.py \
    --train-dir distill_data/train \
    --val-dir   distill_data/val \
    --epochs 20 \
    --run-name orion_student_no_mimic_loss
```

### Decoder-depth ablation (multi-GPU)

Trains four models in parallel across GPUs 0–3, each with a different number
of transformer decoder layers (2, 4, 8, 16):

```bash
cd distill
python train_ablation_decoder_layers.py \
    --train-dir distill_data/train \
    --val-dir   distill_data/val \
    --layers 2,4,8,16 \
    --gpus   0,1,2,3 \
    --use-feature-mimic-loss \
    --epochs 20
```

---

## Step 3 – Evaluate on Bench2Drive220

### Export a single fused checkpoint

The public workflow uses one distilled config and one fused checkpoint.  Fuse
the Orion teacher checkpoint with the distilled student weights once:

```bash
python eval/scripts/export_distill_fused_ckpt.py \
    --orion-ckpt  ckpts/Orion.pth \
    --distill-ckpt distill/results/orion_student_with_mimic_loss/checkpoints/last.pt \
    --out-ckpt   eval/fused_ckpts/orion_student_with_mimic_loss.pth
```

### Open-loop evaluation

```bash
python eval/scripts/evaluate_fused_checkpoint.py \
    --task openloop \
    --config-json distill/results/orion_student_with_mimic_loss/config.json \
    --fused-ckpt eval/fused_ckpts/orion_student_with_mimic_loss.pth \
    --gpus 4 \
    --base-port 29513
```

### Closed-loop evaluation

Preflight on Dev10 routes:

```bash
python eval/scripts/evaluate_fused_checkpoint.py \
    --task closedloop \
    --config-json distill/results/orion_student_with_mimic_loss/config.json \
    --fused-ckpt eval/fused_ckpts/orion_student_with_mimic_loss.pth \
    --mode preflight
```

Full Bench2Drive220 run:

```bash
python eval/scripts/evaluate_fused_checkpoint.py \
    --task closedloop \
    --config-json distill/results/orion_student_with_mimic_loss/config.json \
    --fused-ckpt eval/fused_ckpts/orion_student_with_mimic_loss.pth \
    --mode full
```

Post-process and write `summary.json`:

```bash
python eval/scripts/evaluate_fused_checkpoint.py \
    --task closedloop \
    --config-json distill/results/orion_student_with_mimic_loss/config.json \
    --fused-ckpt eval/fused_ckpts/orion_student_with_mimic_loss.pth \
    --mode postprocess
```

Legacy strict-planner evaluation scripts are still available under `eval/scripts/`
for exact reproduction of older internal workflows.

---

## Reported results

The tables below summarize the reported evaluation numbers used in the paper.
Training-loss summaries are kept in the next section for quick reference.

### Main Bench2Drive220 summary

| Model | Closed-loop DS | Closed-loop SR | Efficiency | Comfortness | Open-loop Avg. L2 | Latency |
|-------|----------------|----------------|------------|-------------|-------------------|---------|
| ORION (0.5B) | 72.9 | 45.8 | - | - | - | - |
| ORION (7B Teacher) | 77.7 | 54.6 | 151.5 | 17.4 | 0.68 | 806 ms |
| Orion-Lite (0.1B) | **80.6** | **55.5** | **157.7** | 10.3 | 0.79 | **267 ms** |

Compared with the 7B teacher, Orion-Lite improves Driving Score by `+2.9`,
Success Rate by `+0.9`, and reduces measured inference latency by about `66.9%`
on an A6000 GPU.

### Ability breakdown

| Model | Merging | Overtaking | Emergency Brake | Give Way | Traffic Sign | Mean |
|-------|---------|------------|-----------------|----------|--------------|------|
| ORION (0.5B) | 26.3 | 62.2 | 55.6 | 50.0 | 63.3 | 51.4 |
| ORION (7B Teacher) | 25.0 | 71.1 | 78.3 | 30.0 | 69.2 | 54.7 |
| Orion-Lite (0.1B) | **28.8** | **75.6** | **78.3** | **50.0** | **70.0** | **60.5** |

### Supervision ablation

| Setting | Closed-loop DS | Closed-loop SR | Ability mean |
|---------|----------------|----------------|--------------|
| Trajectory GT only | 73.9 | 50.0 | 47.7 |
| Feature mimic only | 76.0 | 50.7 | 53.3 |
| Feature mimic + trajectory GT | **80.6** | **55.5** | **60.5** |

The strongest model uses both types of supervision: teacher feature mimic and
trajectory-ground-truth planning losses.

### Training duration comparison

| Setting | Epochs | Closed-loop DS | Closed-loop SR | Ability mean |
|---------|--------|----------------|----------------|--------------|
| Orion | 18 (default) | 77.7 | 54.6 | 54.7 |
| Orion | 24 | 77.1 | 50.7 | 52.0 |
| Orion-Lite | 20 | **80.6** | **55.5** | **60.5** |

### Distance metric ablation (open-loop)

| Distillation loss | L2 avg. | Collision avg. |
|-------------------|---------|----------------|
| L1 | 0.79 | **0.54** |
| L2 | 0.79 | 0.60 |
| KL | 0.79 | 0.63 |
| Huber | **0.75** | 0.70 |

Huber gives the best open-loop L2, while L1 yields the best collision profile.

---

## Distillation training losses

### 6-layer decoder: mimic loss ablation

| Run | Val loss (epoch 20) | Best val loss |
|-----|---------------------|---------------|
| no_mimic  (planning losses only) | 0.1562 | 0.1493 (ep 16) |
| with_mimic (+ L1 feature mimic)  | 0.2895 | 0.2895 (ep 20) |

> Note: the `total` loss is not directly comparable between no-mimic and
> with-mimic runs because the mimic loss term is absent from the no-mimic total.

### Decoder-depth ablation (no mimic loss, `first_try` runs)

Training config: `hidden_dim=1024`, `num_heads=16`, `use_feature_mimic_loss=False`

| Decoder layers | Best val loss |
|----------------|---------------|
| 2  | see `distill/results/ablation_decoder_layers_no_mimic/decoder2l/logs/train.log` |
| 4  | see `distill/results/ablation_decoder_layers_no_mimic/decoder4l/logs/train.log` |
| 8  | see `distill/results/ablation_decoder_layers_no_mimic/decoder8l/logs/train.log` |
| 16 | see `distill/results/ablation_decoder_layers_no_mimic/decoder16l/logs/train.log` |

### Decoder-depth ablation (with mimic loss, `ablation_layers_mimic_no_amp` runs)

Training config: `hidden_dim=1024`, `num_heads=16`, `use_feature_mimic_loss=True`

| Decoder layers | Best val loss (total) |
|----------------|-----------------------|
| 2  | 0.2992 (epoch 17) |
| 4  | see logs |
| 8  | see logs |
| 16 | see logs |

Full training logs available under `distill/results/ablation_decoder_layers_with_mimic/`.

---

## Setup

### 1. Install PyTorch (match your CUDA version)

```bash
# CUDA 11.8
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.4 (newer GPUs / Blackwell)
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### 2. Build the mmcv C extensions

```bash
pip install -v -e . --no-deps
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

The `mmcv/`, `adzoo/`, and `team_code/` directories are included in this repo,
so **no separate Orion installation is needed**.

```
Dependency matrix:
  Step                    | This repo | Orion checkpoint | Bench2Drive | CARLA
  ------------------------|-----------|------------------|-------------|------
  Collect distill data    |     ✓     |        ✓         |             |
  Train student model     |     ✓     |     optional*    |             |
  Export checkpoint       |     ✓     |        ✓         |             |
  Closed-loop eval        |     ✓     |        ✓         |      ✓      |   ✓

* The Orion checkpoint is optional for training: it initialises the VAE weights
  via --orion-ckpt. Training works without it (random VAE init).
```

---

## Citation

```bibtex
@misc{orion-lite,
  title  = {Orion-Lite: Distilling LLM Reasoning into Efficient Vision-Only Driving Models},
  author = {Jing Gu, Niccolò Cavagnero, Gijs Dubbelman },
  year   = {2026},
}
```
