#!/usr/bin/env python
"""Fuse Orion teacher weights and a distilled student checkpoint into one checkpoint.

The resulting checkpoint targets OrionDistilledNew / OrionDistilledAblationLayers:
  - shared Orion detector weights stay at the top level
  - distilled student weights are written under `student_model.*`

This gives us a single distributable checkpoint for open-loop and distilled
closed-loop evaluation.
"""

import argparse
import os
from collections import OrderedDict

import torch


PRUNE_PREFIXES = (
    "lm_head.",
    "present_distribution.",
    "future_distribution.",
    "predict_model.",
    "ego_fut_decoder.",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse Orion.pth and a distilled student checkpoint into one OrionDistilled checkpoint."
    )
    parser.add_argument(
        "--orion-ckpt",
        required=True,
        help="Base Orion checkpoint that provides the shared detector weights.",
    )
    parser.add_argument(
        "--distill-ckpt",
        required=True,
        help="Distilled student checkpoint (.pt) from train_student*.py.",
    )
    parser.add_argument(
        "--out-ckpt",
        required=True,
        help="Output fused checkpoint path.",
    )
    parser.add_argument(
        "--keep-teacher-vae",
        action="store_true",
        help="Keep top-level Orion VAE/LLM keys instead of pruning them from the fused checkpoint.",
    )
    return parser.parse_args()


def _strip_module_prefix(state_dict):
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def _load_state_dict(path, model_state_key=None):
    checkpoint = torch.load(path, map_location="cpu")
    if model_state_key and model_state_key in checkpoint:
        state_dict = checkpoint[model_state_key]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    return checkpoint, _strip_module_prefix(state_dict)


def _validate_orion_state_dict(state_dict, path):
    required_prefixes = ("img_backbone.", "pts_bbox_head.", "map_head.")
    missing = [prefix for prefix in required_prefixes if not any(k.startswith(prefix) for k in state_dict)]
    if missing:
        raise ValueError(
            f"{path} does not look like an Orion detector checkpoint. Missing key groups: {missing}"
        )


def _prune_teacher_only_keys(state_dict):
    pruned = OrderedDict()
    removed = []
    for key, value in state_dict.items():
        if key.startswith(PRUNE_PREFIXES):
            removed.append(key)
            continue
        pruned[key] = value
    return pruned, removed


def main():
    args = parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.out_ckpt))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading Orion checkpoint from: {args.orion_ckpt}", flush=True)
    orion_ckpt, orion_sd = _load_state_dict(args.orion_ckpt)
    _validate_orion_state_dict(orion_sd, args.orion_ckpt)

    if args.keep_teacher_vae:
        fused_sd = OrderedDict(orion_sd)
        removed = []
    else:
        fused_sd, removed = _prune_teacher_only_keys(orion_sd)

    print(f"Loading distilled student checkpoint from: {args.distill_ckpt}", flush=True)
    _, distill_sd = _load_state_dict(args.distill_ckpt)

    replaced = 0
    for key, value in distill_sd.items():
        fused_sd[f"student_model.{key}"] = value
        replaced += 1

    out = {"state_dict": fused_sd}
    meta = dict(orion_ckpt.get("meta", {})) if isinstance(orion_ckpt, dict) else {}
    meta["fused_from_orion_ckpt"] = args.orion_ckpt
    meta["fused_from_distill_ckpt"] = args.distill_ckpt
    meta["student_key_prefix"] = "student_model."
    out["meta"] = meta

    torch.save(out, args.out_ckpt)

    print(f"Pruned teacher-only keys: {len(removed)}", flush=True)
    print(f"Injected distilled student keys: {replaced}", flush=True)
    print(f"Saved fused checkpoint to: {args.out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
