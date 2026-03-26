"""
Export a distilled student decoder into a full OrionStudentPlanner checkpoint.

The distillation training scripts (train_student_with_orion_loss.py,
train_ablation_decoder_layers.py) only save the student planning decoder
weights (OrionMimicModel).  The Bench2Drive evaluation pipeline requires a
full OrionStudentPlanner checkpoint that also includes the vision backbone and
detection/map head weights.

This script:
  1. Loads a base OrionStudentPlanner checkpoint (e.g. from exp9 stage2).
  2. Replaces the student decoder weights with the distilled decoder weights.
  3. Saves the merged checkpoint to the specified output path.

Usage:
    python eval/scripts/export_distill_ckpt.py \\
        --base-ckpt  /path/to/Orion/adzoo/orion/work_dirs/.../latest.pth \\
        --distill-ckpt /path/to/Orion-Lite/distill/results/<run>/checkpoints/best.pt \\
        --out-ckpt   /path/to/output/merged.pth \\
        [--num-layers 2]

Arguments:
    --base-ckpt    Full OrionStudentPlanner checkpoint (backbone + heads + decoder).
    --distill-ckpt Distill training checkpoint produced by train_student_with_orion_loss.py.
    --out-ckpt     Output path for the merged checkpoint.
    --num-layers   Number of decoder transformer layers in the distilled model.
                   Used only for verification; must match the distilled model config.
"""

import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge distilled decoder weights into a full OrionStudentPlanner checkpoint.")
    parser.add_argument("--base-ckpt",    required=True,
                        help="Base OrionStudentPlanner checkpoint (backbone + heads + decoder).")
    parser.add_argument("--distill-ckpt", required=True,
                        help="Distilled decoder checkpoint (OrionMimicModel, from train_student_with_orion_loss.py).")
    parser.add_argument("--out-ckpt",     required=True,
                        help="Output path for the merged checkpoint.")
    parser.add_argument("--num-layers",   type=int, default=None,
                        help="Expected number of decoder layers (for verification only).")
    return parser.parse_args()


def _strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def _validate_base_checkpoint(base_sd, base_path):
    required_prefixes = [
        "student_model.",
        "present_distribution.",
        "future_distribution.",
        "predict_model.",
        "ego_fut_decoder.",
        "img_backbone.",
        "pts_bbox_head.",
        "map_head.",
    ]
    missing = [prefix for prefix in required_prefixes if not any(k.startswith(prefix) for k in base_sd)]
    if missing:
        raise ValueError(
            "Base checkpoint does not look like a full OrionStudentPlanner stage-2 checkpoint. "
            f"Missing key groups: {missing}. "
            f"Received: {base_path}. "
            "This usually means you passed Orion.pth or another teacher checkpoint instead of the "
            "student-planner stage-2 checkpoint."
        )


def _resolve_base_key(base_sd, clean_key):
    candidates = []
    direct_candidates = [
        clean_key,
        f"student_model.{clean_key}",
        f"student_decoder.{clean_key}",
    ]
    for candidate in direct_candidates:
        if candidate in base_sd and candidate not in candidates:
            candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous direct mapping for distill key '{clean_key}': {candidates}"
        )

    suffix_matches = [bk for bk in base_sd if bk.endswith(clean_key)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise RuntimeError(
            f"Ambiguous suffix mapping for distill key '{clean_key}': {suffix_matches}"
        )
    return None


def main():
    args = parse_args()

    print(f"Loading base checkpoint from: {args.base_ckpt}")
    base = torch.load(args.base_ckpt, map_location="cpu")
    base_sd = base.get("state_dict", base)
    _validate_base_checkpoint(base_sd, args.base_ckpt)

    print(f"Loading distill checkpoint from: {args.distill_ckpt}")
    distill = torch.load(args.distill_ckpt, map_location="cpu")
    distill_sd = _strip_module_prefix(distill.get("model_state_dict", distill))

    # Distillation checkpoints may come from:
    # 1. `train_student.py` -> OrionStudent keys like `transformer_decoder...`
    # 2. `train_student_with_orion_loss.py` -> OrionMimicModel keys like
    #    `transformer_decoder...` plus `present_distribution...`, etc.
    #
    # The full closed-loop planner checkpoint stores the decoder under
    # `student_model.*`, while the VAE/planning heads live at the top level.
    decoder_keys_in_base = [k for k in base_sd if k.startswith("student_model.")]
    decoder_keys_in_distill = list(distill_sd.keys())

    print(f"  Base checkpoint student decoder keys: {len(decoder_keys_in_base)}")
    print(f"  Distill checkpoint keys:              {len(decoder_keys_in_distill)}")

    merged_sd = dict(base_sd)
    replaced = 0
    replaced_groups = {
        "student_model": 0,
        "present_distribution": 0,
        "future_distribution": 0,
        "predict_model": 0,
        "ego_fut_decoder": 0,
        "other": 0,
    }
    for clean, dv in distill_sd.items():
        target_key = _resolve_base_key(base_sd, clean)
        if target_key is not None:
            merged_sd[target_key] = dv
            replaced += 1
            group = target_key.split(".", 1)[0]
            if group not in replaced_groups:
                group = "other"
            replaced_groups[group] += 1
        else:
            print(f"  [WARN] No match for distill key '{clean}' in base checkpoint – skipping.")

    print(f"\nReplaced {replaced}/{len(decoder_keys_in_distill)} distill keys in base checkpoint.")
    print("Replacement breakdown:")
    for group, count in replaced_groups.items():
        print(f"  {group}: {count}")

    if replaced == 0:
        raise RuntimeError(
            "Did not replace any keys from the distilled checkpoint. "
            "Please verify that the distill checkpoint and base checkpoint are compatible."
        )

    if args.num_layers is not None:
        # Sanity-check: count how many decoder layer blocks exist in the merged student model.
        layer_keys = [
            k for k in merged_sd
            if k.startswith("student_model.transformer_decoder.layers.") and ".layers." in k
        ]
        layer_indices = set()
        for k in layer_keys:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    layer_indices.add(parts[i + 1])
        print(f"Detected decoder layer indices in merged checkpoint: {sorted(layer_indices)}")
        if len(layer_indices) != args.num_layers:
            print(f"[WARN] Expected {args.num_layers} layers, found {len(layer_indices)}.")

    out = {"state_dict": merged_sd}
    if "meta" in base:
        out["meta"] = base["meta"]

    torch.save(out, args.out_ckpt)
    print(f"\nSaved merged checkpoint to: {args.out_ckpt}")


if __name__ == "__main__":
    main()
