#!/usr/bin/env python
"""Launch Orion-Lite open-loop evaluation for a distilled run."""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime

from evaluate_distill_result import (
    DEFAULT_ORION_ROOT,
    REPO_ROOT,
    artifact_stem_for_paths,
    build_student_env,
    default_fused_ckpt_path,
    load_config,
    resolve_distill_ckpt,
    resolve_orion_ckpt,
    resolve_result_dir,
)


DEFAULT_LOG_ROOT = os.path.join(REPO_ROOT, "test", "orion_student_openloop_distill")
DEFAULT_GPUS = 1
DEFAULT_BASE_PORT = 29503


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Open-loop evaluation wrapper for a fused OrionDistilled checkpoint. "
            "For the public fused-checkpoint workflow, prefer "
            "`evaluate_fused_checkpoint.py`."
        )
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Run directory under distill/results, or an absolute path to one.",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Path to a run config.json. Use this with --fused-ckpt to evaluate from just one checkpoint and one config.",
    )
    parser.add_argument(
        "--orion-root",
        default=DEFAULT_ORION_ROOT,
        help="Original Orion repository root used for data defaults and fallback distilled checkpoints.",
    )
    parser.add_argument(
        "--orion-ckpt",
        default=None,
        help="Base Orion checkpoint used to build the fused checkpoint. Defaults to Orion-Lite/ckpts/Orion.pth if present.",
    )
    parser.add_argument(
        "--distill-ckpt",
        default=None,
        help="Override distilled checkpoint path directly.",
    )
    parser.add_argument(
        "--fused-ckpt",
        default=None,
        help=(
            "Path for the fused OrionDistilled checkpoint. "
            "If the file already exists, it will be reused unless --rebuild-fused is set."
        ),
    )
    parser.add_argument(
        "--rebuild-fused",
        action="store_true",
        help="Force re-export of the fused checkpoint even if --fused-ckpt already exists.",
    )
    parser.add_argument(
        "--which",
        choices=["best", "last"],
        default="best",
        help="Which checkpoint under <result-dir>/checkpoints to use when --distill-ckpt is not set.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional explicit open-loop config. Defaults to the matching fused distilled config.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root. Defaults to <orion-root>/data/bench2drive.",
    )
    parser.add_argument(
        "--info-root",
        default=None,
        help="Info root. Defaults to <orion-root>/data/infos.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=DEFAULT_GPUS,
        help="Number of GPUs for distributed open-loop evaluation.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=DEFAULT_BASE_PORT,
        help="Master port passed to the distributed evaluator.",
    )
    parser.add_argument(
        "--log-root",
        default=DEFAULT_LOG_ROOT,
        help="Directory where wrapper stdout logs and summaries will be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved evaluation command without running it.",
    )
    return parser.parse_args()


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _extract_metrics(text):
    metrics = {}
    for key in (
        "plan_L2_1s",
        "plan_L2_2s",
        "plan_L2_3s",
        "plan_obj_col_1s",
        "plan_obj_col_2s",
        "plan_obj_col_3s",
        "plan_obj_box_col_1s",
        "plan_obj_box_col_2s",
        "plan_obj_box_col_3s",
        "mAP",
        "mATE",
        "mASE",
        "mAOE",
        "mAVE",
        "NDS",
    ):
        match = re.search(rf"^{re.escape(key)}:\s*([0-9.]+)\s*$", text, flags=re.MULTILINE)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def _capture_existing_children(path):
    if not os.path.isdir(path):
        return set()
    return {entry.name for entry in os.scandir(path) if entry.is_dir()}


def _discover_new_eval_dir(base_dir, before_children):
    if not os.path.isdir(base_dir):
        return None

    after_entries = [entry for entry in os.scandir(base_dir) if entry.is_dir()]
    new_entries = [entry.path for entry in after_entries if entry.name not in before_children]
    if new_entries:
        new_entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return new_entries[0]

    after_entries.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
    return after_entries[0].path if after_entries else None


def _resolve_openloop_config(override, config):
    if override:
        path = os.path.abspath(override)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Open-loop config not found: {path}")
        return path

    num_layers = int(config["num_layers"])
    configs_root = os.path.join(REPO_ROOT, "adzoo", "orion", "configs")
    if num_layers == 6:
        path = os.path.join(configs_root, "orion_stage3_infer_distill_fused.py")
    elif num_layers in {2, 4, 8, 16}:
        path = os.path.join(
            configs_root,
            "ablation_layers",
            "orion_stage3_infer_distill_ablation_fused.py",
        )
    else:
        raise ValueError(
            f"Unsupported num_layers={num_layers}. Expected 6, 2, 4, 8, or 16."
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Open-loop config not found: {path}")
    return path


def _build_cfg_options(data_root, info_root):
    options = [
        f"data.test.data_root={os.path.abspath(data_root)}",
        f"data.test.ann_file={os.path.join(os.path.abspath(info_root), 'b2d_infos_val.pkl')}",
        f"data.test.map_root={os.path.join(os.path.abspath(data_root), 'maps')}",
        f"data.test.map_file={os.path.join(os.path.abspath(info_root), 'b2d_map_infos.pkl')}",
    ]
    return options


def _run_and_log(cmd, cwd, env, stdout_path, dry_run=False):
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return

    with open(stdout_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Open-loop evaluation failed. See log: {stdout_path}")


def main():
    args = parse_args()

    result_dir = resolve_result_dir(args.result_dir) if args.result_dir else None
    result_config, config_path, result_dir = load_config(result_dir=result_dir, config_json=args.config_json)
    orion_root = os.path.abspath(args.orion_root)
    orion_ckpt = resolve_orion_ckpt(args.orion_ckpt, orion_root)
    fused_ckpt = os.path.abspath(args.fused_ckpt or default_fused_ckpt_path(result_dir=result_dir, config_path=config_path))
    reuse_existing_fused = os.path.isfile(fused_ckpt) and not args.rebuild_fused
    distill_ckpt = None
    if not reuse_existing_fused:
        distill_ckpt = resolve_distill_ckpt(
            result_dir,
            args.distill_ckpt,
            which=args.which,
            config=result_config,
            orion_root=orion_root,
        )
    openloop_config = _resolve_openloop_config(args.config, result_config)

    data_root = os.path.abspath(args.data_root or os.path.join(orion_root, "data", "bench2drive"))
    info_root = os.path.abspath(args.info_root or os.path.join(orion_root, "data", "infos"))
    ann_file = os.path.join(info_root, "b2d_infos_val.pkl")
    map_file = os.path.join(info_root, "b2d_map_infos.pkl")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"Validation info file not found: {ann_file}")
    if not os.path.isfile(map_file):
        raise FileNotFoundError(f"Map info file not found: {map_file}")

    os.makedirs(os.path.dirname(fused_ckpt), exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = _slug(artifact_stem_for_paths(result_dir=result_dir, config_path=config_path))
    log_dir = os.path.join(os.path.abspath(args.log_root), timestamp)
    os.makedirs(log_dir, exist_ok=True)
    stdout_path = os.path.join(log_dir, f"openloop_{run_tag}_{args.which}.txt")
    summary_path = os.path.join(log_dir, f"summary_{run_tag}_{args.which}.json")

    config_basename = os.path.splitext(os.path.basename(openloop_config))[0]
    test_output_base = os.path.join(REPO_ROOT, "test", config_basename)
    before_children = _capture_existing_children(test_output_base)

    cfg_options = _build_cfg_options(data_root, info_root)
    student_env = build_student_env(result_config)

    print(f"config_json    : {config_path}")
    print(f"fused_ckpt     : {fused_ckpt}")
    print(f"reuse_fused    : {reuse_existing_fused}")
    print(f"openloop_config: {openloop_config}")
    print(f"data_root      : {data_root}")
    print(f"info_root      : {info_root}")
    print(f"num_layers     : {result_config['num_layers']}")
    print(f"use_mimic      : {bool(result_config.get('use_feature_mimic_loss', False))}")
    print(f"gpus           : {args.gpus}")
    print(f"log_dir        : {log_dir}")
    print(f"stdout_log     : {stdout_path}")
    print(f"summary_json   : {summary_path}")
    if not reuse_existing_fused:
        print(f"orion_ckpt     : {orion_ckpt}")
        print(f"distill_ckpt   : {distill_ckpt}")

    export_cmd = None
    if reuse_existing_fused:
        print(f"Using existing fused checkpoint: {fused_ckpt}")
    else:
        export_cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "eval", "scripts", "export_distill_fused_ckpt.py"),
            "--orion-ckpt",
            orion_ckpt,
            "--distill-ckpt",
            distill_ckpt,
            "--out-ckpt",
            fused_ckpt,
        ]
        _run_and_log(export_cmd, REPO_ROOT, os.environ.copy(), stdout_path, dry_run=args.dry_run)

    eval_cmd = [
        "bash",
        os.path.join(REPO_ROOT, "adzoo", "orion", "orion_dist_eval.sh"),
        openloop_config,
        fused_ckpt,
        str(args.gpus),
        "--cfg-options",
        *cfg_options,
    ]
    eval_env = os.environ.copy()
    eval_env.update(student_env)
    eval_env["PORT"] = str(args.base_port)

    if args.dry_run:
        if reuse_existing_fused:
            print(f"# using existing fused checkpoint: {fused_ckpt}")
        print("$ " + " ".join(shlex.quote(c) for c in eval_cmd))
        return

    with open(stdout_path, "a", encoding="utf-8") as f:
        if export_cmd is not None:
            f.write("$ " + " ".join(shlex.quote(c) for c in export_cmd) + "\n")
        f.write("$ " + " ".join(shlex.quote(c) for c in eval_cmd) + "\n")
        proc = subprocess.run(
            eval_cmd,
            cwd=REPO_ROOT,
            env=eval_env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Open-loop evaluation failed. See log: {stdout_path}")

    eval_output_dir = _discover_new_eval_dir(test_output_base, before_children)

    summary = {
        "source_run_dir": result_dir,
        "result_config": config_path,
        "distill_ckpt": distill_ckpt,
        "orion_ckpt": orion_ckpt,
        "fused_ckpt": fused_ckpt,
        "which": args.which,
        "openloop_config": openloop_config,
        "stdout_log": stdout_path,
        "test_output_dir": eval_output_dir,
        "metrics": _extract_metrics(open(stdout_path, "r", encoding="utf-8").read()),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Open-loop log     : {stdout_path}")
    print(f"Open-loop summary : {summary_path}")
    if eval_output_dir:
        print(f"test.py output dir: {eval_output_dir}")


if __name__ == "__main__":
    main()
