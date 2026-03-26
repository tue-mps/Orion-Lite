#!/usr/bin/env python
"""Public evaluation entrypoint for fused Orion-Lite distilled checkpoints."""

import argparse
import os
import shlex
import subprocess
import sys

from evaluate_distill_result import DEFAULT_B2D_ROOT, DEFAULT_ORION_ROOT, REPO_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fused Orion-Lite distilled checkpoint using only one "
            "config.json and one fused checkpoint. This is the public-facing "
            "entrypoint; it always uses the fused distilled backend."
        )
    )
    parser.add_argument(
        "--task",
        choices=["openloop", "closedloop"],
        required=True,
        help="Which evaluation workflow to run.",
    )
    parser.add_argument(
        "--config-json",
        required=True,
        help="Path to the distilled run config.json.",
    )
    parser.add_argument(
        "--fused-ckpt",
        required=True,
        help="Path to the fused OrionDistilled checkpoint.",
    )
    parser.add_argument(
        "--mode",
        choices=["preflight", "full", "postprocess", "all"],
        default="preflight",
        help="Closed-loop stage to run. Ignored for open-loop.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Open-loop GPU count. Ignored for closed-loop.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=29503,
        help="Open-loop distributed master port. Ignored for closed-loop.",
    )
    parser.add_argument(
        "--orion-root",
        default=DEFAULT_ORION_ROOT,
        help="Original Orion repository root used for open-loop data defaults.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional open-loop dataset root override.",
    )
    parser.add_argument(
        "--info-root",
        default=None,
        help="Optional open-loop info root override.",
    )
    parser.add_argument(
        "--log-root",
        default=None,
        help="Optional open-loop wrapper log directory override.",
    )
    parser.add_argument(
        "--b2d-root",
        default=DEFAULT_B2D_ROOT,
        help="Bench2Drive repository root for closed-loop evaluation.",
    )
    parser.add_argument(
        "--reference-summary",
        default=None,
        help="Optional reference summary.json used by the closed-loop wrapper after postprocess/all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without running it.",
    )
    return parser.parse_args()


def _run_command(cmd, dry_run=False):
    print("$ " + " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main():
    args = parse_args()

    config_json = os.path.abspath(args.config_json)
    fused_ckpt = os.path.abspath(args.fused_ckpt)

    if not os.path.isfile(config_json):
        raise FileNotFoundError(f"Missing config.json: {config_json}")
    if not os.path.isfile(fused_ckpt):
        raise FileNotFoundError(f"Missing fused checkpoint: {fused_ckpt}")

    if args.task == "openloop":
        script = os.path.join(REPO_ROOT, "eval", "scripts", "evaluate_distill_result_openloop.py")
        cmd = [
            sys.executable,
            script,
            "--config-json",
            config_json,
            "--fused-ckpt",
            fused_ckpt,
            "--orion-root",
            os.path.abspath(args.orion_root),
            "--gpus",
            str(args.gpus),
            "--base-port",
            str(args.base_port),
        ]
        if args.data_root:
            cmd.extend(["--data-root", os.path.abspath(args.data_root)])
        if args.info_root:
            cmd.extend(["--info-root", os.path.abspath(args.info_root)])
        if args.log_root:
            cmd.extend(["--log-root", os.path.abspath(args.log_root)])
    else:
        script = os.path.join(REPO_ROOT, "eval", "scripts", "evaluate_distill_result.py")
        cmd = [
            sys.executable,
            script,
            "--config-json",
            config_json,
            "--fused-ckpt",
            fused_ckpt,
            "--style",
            "fused_distilled",
            "--mode",
            args.mode,
            "--b2d-root",
            os.path.abspath(args.b2d_root),
        ]
        if args.reference_summary:
            cmd.extend(["--reference-summary", os.path.abspath(args.reference_summary)])

    if args.dry_run:
        cmd.append("--dry-run")

    print(f"task          : {args.task}")
    print(f"config_json   : {config_json}")
    print(f"fused_ckpt    : {fused_ckpt}")
    if args.task == "openloop":
        print(f"gpus          : {args.gpus}")
        print(f"base_port     : {args.base_port}")
    else:
        print(f"mode          : {args.mode}")
        print(f"b2d_root      : {os.path.abspath(args.b2d_root)}")

    _run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
