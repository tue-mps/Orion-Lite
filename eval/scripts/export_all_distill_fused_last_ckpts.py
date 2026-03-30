#!/usr/bin/env python
"""Batch-export fused checkpoints for every distillation run that has last.pt.

This scans distill/results recursively for:
  */checkpoints/last.pt

and creates a fused checkpoint for each run using the same output naming
convention as the evaluation wrappers:
  eval/fused_ckpts/<result_dir_relative_to_distill_results>.pth

Examples:
  python eval/scripts/export_all_distill_fused_last_ckpts.py
  python eval/scripts/export_all_distill_fused_last_ckpts.py --force
  python eval/scripts/export_all_distill_fused_last_ckpts.py --dry-run
"""

import argparse
import os
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RESULTS_ROOT = os.path.join(REPO_ROOT, "distill", "results")
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, "eval", "fused_ckpts")
DEFAULT_ORION_CKPT = os.path.join(REPO_ROOT, "ckpts", "Orion.pth")
EXPORTER = os.path.join(REPO_ROOT, "eval", "scripts", "export_distill_fused_ckpt.py")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse every distillation last.pt under distill/results into eval/fused_ckpts."
    )
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory that contains distillation result folders.",
    )
    parser.add_argument(
        "--orion-ckpt",
        default=DEFAULT_ORION_CKPT,
        help="Base Orion checkpoint used for all fusions.",
    )
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help="Directory where fused checkpoints will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite fused checkpoints even if they already exist.",
    )
    parser.add_argument(
        "--keep-teacher-vae",
        action="store_true",
        help="Pass --keep-teacher-vae through to the single-checkpoint exporter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    return parser.parse_args()


def artifact_stem_for_result_dir(result_dir, results_root):
    rel = os.path.relpath(os.path.abspath(result_dir), os.path.abspath(results_root))
    raw = rel if not rel.startswith("..") else os.path.basename(os.path.abspath(result_dir))
    return raw.replace(os.sep, "__")


def find_last_checkpoints(results_root):
    hits = []
    for root, _, files in os.walk(results_root):
        if "last.pt" not in files:
            continue
        if os.path.basename(root) != "checkpoints":
            continue
        hits.append(os.path.join(root, "last.pt"))
    return sorted(hits)


def main():
    args = parse_args()
    results_root = os.path.abspath(args.results_root)
    out_root = os.path.abspath(args.out_root)
    orion_ckpt = os.path.abspath(args.orion_ckpt)

    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Missing results root: {results_root}")
    if not os.path.isfile(orion_ckpt):
        raise FileNotFoundError(f"Missing Orion checkpoint: {orion_ckpt}")
    if not os.path.isfile(EXPORTER):
        raise FileNotFoundError(f"Missing exporter script: {EXPORTER}")

    last_ckpts = find_last_checkpoints(results_root)
    if not last_ckpts:
        raise FileNotFoundError(f"No last.pt checkpoints found under: {results_root}")

    os.makedirs(out_root, exist_ok=True)

    print(f"results_root : {results_root}")
    print(f"orion_ckpt   : {orion_ckpt}")
    print(f"out_root     : {out_root}")
    print(f"runs_found    : {len(last_ckpts)}")

    built = 0
    skipped = 0

    for distill_ckpt in last_ckpts:
        result_dir = os.path.dirname(os.path.dirname(distill_ckpt))
        stem = artifact_stem_for_result_dir(result_dir, results_root)
        out_ckpt = os.path.join(out_root, f"{stem}.pth")

        print("")
        print(f"result_dir   : {result_dir}")
        print(f"distill_ckpt : {distill_ckpt}")
        print(f"fused_ckpt   : {out_ckpt}")

        if os.path.isfile(out_ckpt) and not args.force:
            print("status       : skip (exists)")
            skipped += 1
            continue

        cmd = [
            sys.executable,
            EXPORTER,
            "--orion-ckpt",
            orion_ckpt,
            "--distill-ckpt",
            distill_ckpt,
            "--out-ckpt",
            out_ckpt,
        ]
        if args.keep_teacher_vae:
            cmd.append("--keep-teacher-vae")

        print("status       : build")
        print("$ " + " ".join(cmd))

        if not args.dry_run:
            subprocess.run(cmd, check=True)
        built += 1

    print("")
    print(f"done         : built={built} skipped={skipped} total={len(last_ckpts)}")


if __name__ == "__main__":
    main()
