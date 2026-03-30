<!-- # Orion-Lite: Distilling LLM Reasoning into Efficient Vision-Only Driving Models -->

<div align="center">
<h3>[CVPRW 26] Orion-Lite: Distilling LLM Reasoning into Efficient Vision-Only Driving Models</h3>
Jing Gu, Cavagnero, Niccolò, Dubbelman, Gijs<sup>†</sup>

Eindhoven University of Technology, MPS Lab

(†) Team leader.
</div>

## Abstract

Leveraging the general world knowledge of Large Language Models (LLMs) holds significant promise for improving the ability of autonomous driving systems to handle rare and complex scenarios. While integrating LLMs into Vision-Language-Action (VLA) models has yielded state-of-the-art performance, their massive parameter counts pose severe challenges for latency-sensitive and energy-efficient deployment. Distilling LLM knowledge into a compact driving model offers a compelling solution to retain these reasoning capabilities while maintaining a manageable computational footprint. Although previous works have demonstrated the efficacy of distillation, these efforts have primarily focused on relatively simple scenarios and open-loop evaluations. Therefore, in this work, we investigate LLM distillation in more complex, interactive scenarios under closed-loop evaluation. We demonstrate that through a combination of latent feature distillation and ground-truth trajectory supervision, an efficient vision-only student model \textbf{Orion-Lite} can even surpass the performance of its massive VLA teacher, ORION. Setting a new state-of-the-art on the rigorous Bench2Drive benchmark, with a Driving Score of 80.6. Ultimately, this reveals that vision-only architectures still possess significant, untapped potential for high-performance reactive planning.

## Overview

<p align="center">
  <a href="assets/figures/paper_figs.pdf">
    <img src="assets/figures/paper_figs.png" alt="Orion-Lite overview figure" width="100%">
  </a>
</p>

The student decoder takes teacher visual queries as input and predicts the ego planning feature directly, bypassing the expensive LLM forward pass at inference time. The repo supports distilled training, fused checkpoint export, and Bench2Drive-compatible evaluation.

## Links

- Paper:
- Project page: [MPS-LAB](https://www.tue-mps.org/)
- Checkpoints: [Orion-Lite](https://huggingface.co/JG-GJ/Orion-Lite/tree/main/fused_ckpts)
- Evaluation JSON: Orion-Lite/eval/eval_results

## Currently Supported Features

- [√] Teacher distillation-data collection to `.npz`
- [√] Student decoder training with regression-only losses
- [√] Student training with Orion planning, boundary, collision, and feature-mimic losses
- [√] Fused checkpoint export for public evaluation
- [√] Open-loop evaluation
- [√] Closed-loop Bench2Drive evaluation

## Getting Started

```bash
git clone https://github.com/tue-mps/Orion-Lite.git
cd Orion-Lite
conda create -n orion-lite python=3.8 -y
conda activate orion-lite

# Pick the PyTorch build that matches your CUDA setup.
# CUDA 11.8
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
# pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install -v -e .
pip install -r requirements.txt
```

## Distillation Data Collection

Orion-Lite provides a dedicated collection pipeline in [data_collection/README.md](data_collection/README.md). The collection path uses `OrionCollect`, so you do not need to patch `orion.py` manually.

Collect the full training and validation splits:

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

Each frame is saved as a compressed `.npz` file containing the visual queries, teacher planning token, and planning supervision used during student training.

## Train

### Full Training with Orion Planning Losses

Train `OrionMimicModel` with planning regulation, boundary, collision, and optional feature-mimic losses:

```bash
cd distill
python train_student_with_orion_loss.py \
    --train-dir ../distill_data/train \
    --val-dir ../distill_data/val \
    --orion-ckpt ../ckpts/Orion.pth \
    --use-feature-mimic-loss \
    --epochs 20 \
    --run-name orion_student_with_mimic_loss
```

The distilled run is written under `distill/runs/<run_name>/`.

## Evaluation

The public evaluation flow uses only:

- one distilled run `config.json` from `distill/results/`
- one fused checkpoint from `eval/fused_ckpts/`

For the main Orion-Lite model, use:

- `distill/results/orion_student_with_mimic_loss/config.json`
- `eval/fused_ckpts/orion_lite.pth`

### Open-Loop Evaluation

```bash
python eval/scripts/evaluate_fused_checkpoint.py \
    --task openloop \
    --config-json distill/results/orion_student_with_mimic_loss/config.json \
    --fused-ckpt eval/fused_ckpts/orion_lite.pth \
    --gpus 4 \
    --base-port 29513
```

`--data-root`, `--info-root`, and `--log-root` can be used to override defaults when needed.

### Closed-Loop Evaluation

Full Bench2Drive220 evaluation:

```bash
python eval/scripts/evaluate_fused_checkpoint.py \
    --task closedloop \
    --config-json distill/results/orion_student_with_mimic_loss/config.json \
    --fused-ckpt eval/fused_ckpts/orion_lite.pth \
    --mode full
```

## Results and Checkpoints

### Main Bench2Drive220 Summary

| Model | Closed-loop DS | Closed-loop SR | Efficiency | Comfortness | Open-loop Avg. L2 | Latency | Config | Checkpoint | Eval JSON |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| ORION (0.5B) | 72.9 | 45.8 | - | - | - | - |  |  |  |
| ORION (7B Teacher) | 77.7 | 54.6 | 151.5 | 17.4 | 0.68 | 806 ms | [orion_stage3.py](adzoo/orion/configs/orion_stage3.py) |  |  |
| Orion-Lite (0.1B) | **80.6** | **55.5** | **157.7** | 10.3 | 0.79 | **267 ms** |  |  |  |

Compared with the 7B teacher, Orion-Lite improves Driving Score by `+2.9`, Success Rate by `+0.9`.

### Ability Breakdown

| Model | Merging | Overtaking | Emergency Brake | Give Way | Traffic Sign | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORION (0.5B) | 26.3 | 62.2 | 55.6 | 50.0 | 63.3 | 51.4 |
| ORION (7B Teacher) | 25.0 | 71.1 | 78.3 | 30.0 | 69.2 | 54.7 |
| Orion-Lite (0.1B) | **28.8** | **75.6** | **78.3** | **50.0** | **70.0** | **60.5** |

### Supervision Ablation

| Setting | Closed-loop DS | Closed-loop SR | Ability Mean |
| :--- | ---: | ---: | ---: |
| Trajectory GT only | 73.9 | 50.0 | 47.7 |
| Feature mimic only | 76.0 | 50.7 | 53.3 |
| Feature mimic + trajectory GT | **80.6** | **55.5** | **60.5** |

### Training Duration Comparison

| Setting | Epochs | Closed-loop DS | Closed-loop SR | Ability Mean |
| :--- | ---: | ---: | ---: | ---: |
| ORION | 18 (default) | 77.7 | 54.6 | 54.7 |
| ORION | 24 | 77.1 | 50.7 | 52.0 |
| Orion-Lite | 20 | **80.6** | **55.5** | **60.5** |

### Distillation Loss Ablation

| Distillation loss | Open-loop Avg. L2 | Collision Avg. |
| :--- | ---: | ---: |
| L1 | 0.79 | **0.54** |
| L2 | 0.79 | 0.60 |
| KL | 0.79 | 0.63 |
| Huber | **0.75** | 0.70 |

Huber gives the best open-loop L2, while L1 yields the best collision profile in this ablation.

## Citation

```bibtex

```
