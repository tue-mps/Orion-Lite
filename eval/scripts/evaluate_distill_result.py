#!/usr/bin/env python
"""Launch fused closed-loop evaluation for Orion-Lite."""

import argparse
import json
import os
import subprocess


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RESULTS_ROOT = os.path.join(REPO_ROOT, "distill", "results")
DEFAULT_FUSED_ROOT = os.path.join(REPO_ROOT, "eval", "fused_ckpts")
DEFAULT_B2D_ROOT = "/mnt/adas7tb/jgu/Bench2Drive"
DEFAULT_ORION_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "Orion"))
DEFAULT_PLANNER_TYPE = "only_traj"
DEFAULT_BASE_CHECKPOINT_ENDPOINT = "eval_bench2drive220"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop evaluation wrapper for Orion-Lite fused checkpoints. "
            "Use this with one distilled run config.json and one fused checkpoint."
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
        help="Path to a distilled run config.json.",
    )
    parser.add_argument(
        "--fused-ckpt",
        default=None,
        help=(
            "Path to the fused Orion-Lite checkpoint. "
            "Defaults to eval/fused_ckpts/<artifact>.pth when omitted."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["preflight", "full", "postprocess", "all"],
        default="preflight",
        help="Closed-loop evaluation stage to run.",
    )
    parser.add_argument(
        "--b2d-root",
        default=DEFAULT_B2D_ROOT,
        help="Bench2Drive repository root.",
    )
    parser.add_argument(
        "--reference-summary",
        default=None,
        help="Optional Orion-side summary.json to compare against after postprocess/all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved export/eval commands without running them.",
    )
    return parser.parse_args()


def resolve_result_dir(path):
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(DEFAULT_RESULTS_ROOT, path))


def load_config(result_dir=None, config_json=None):
    if config_json:
        config_path = os.path.abspath(config_json)
        result_dir = os.path.dirname(config_path)
    elif result_dir:
        config_path = os.path.join(result_dir, "config.json")
    else:
        raise ValueError("Provide either --result-dir or --config-json.")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f), config_path, result_dir


def artifact_stem_for_paths(result_dir=None, config_path=None):
    if result_dir:
        rel = os.path.relpath(os.path.abspath(result_dir), DEFAULT_RESULTS_ROOT)
        raw = rel if not rel.startswith("..") else os.path.basename(os.path.abspath(result_dir))
    elif config_path:
        cfg_dir = os.path.dirname(os.path.abspath(config_path))
        rel = os.path.relpath(cfg_dir, DEFAULT_RESULTS_ROOT)
        if rel.startswith(".."):
            parts = [p for p in cfg_dir.rstrip(os.sep).split(os.sep) if p]
            raw = "__".join(parts[-2:]) if len(parts) >= 2 else os.path.basename(cfg_dir)
        else:
            raw = rel
    else:
        raise ValueError("artifact_stem_for_paths requires result_dir or config_path.")
    return raw.replace(os.sep, "__")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_fused_closedloop_target(config):
    num_layers = int(config["num_layers"])
    if num_layers == 6:
        return {
            "family": "6l_distilled",
            "script": os.path.join(REPO_ROOT, "eval", "scripts", "run_orion_distilled_fused_multi.sh"),
            "config": os.path.join(REPO_ROOT, "configs", "orion_lite_closedloop.py"),
            "mimic_mode": "with_mimic" if bool(config.get("use_feature_mimic_loss", False)) else "no_mimic",
        }
    raise ValueError(
        f"Unsupported num_layers={num_layers}. "
        "This simplified Orion-Lite repo keeps the 6-layer fused closed-loop path only."
    )


def infer_algo_name(config, target):
    use_feature_mimic_loss = bool(config.get("use_feature_mimic_loss", False))
    suffix = "with_mimic" if use_feature_mimic_loss else "no_mimic"
    return f"orion_distilled_6l_{suffix}_fused"


def resolve_eval_output_paths(config, target, b2d_root, env=None):
    env = env or os.environ
    b2d_root = os.path.abspath(b2d_root)
    algo = infer_algo_name(config, target)
    planner_type = env.get("PLANNER_TYPE", DEFAULT_PLANNER_TYPE)
    base_checkpoint_endpoint = env.get("BASE_CHECKPOINT_ENDPOINT", DEFAULT_BASE_CHECKPOINT_ENDPOINT)
    result_dir = os.path.join(b2d_root, f"{algo}_b2d_{planner_type}")

    save_path = env.get("SAVE_PATH", f"./eval_bench2drive220_{algo}_{planner_type}")
    if not os.path.isabs(save_path):
        save_path = os.path.join(b2d_root, save_path)

    return {
        "algo": algo,
        "planner_type": planner_type,
        "result_dir": result_dir,
        "preflight_json": os.path.join(result_dir, f"{base_checkpoint_endpoint}_dev10.json"),
        "split_json_prefix": os.path.join(result_dir, f"{base_checkpoint_endpoint}_"),
        "merged_json": os.path.join(result_dir, "merged.json"),
        "summary_json": os.path.join(result_dir, "summary.json"),
        "save_path": save_path,
    }

def default_fused_ckpt_path(result_dir=None, config_path=None):
    stem = artifact_stem_for_paths(result_dir=result_dir, config_path=config_path)
    return os.path.join(DEFAULT_FUSED_ROOT, f"{stem}.pth")


def build_student_env(config):
    env = {
        "ORION_STUDENT_INPUT_DIM": str(config.get("input_dim", 4096)),
        "ORION_STUDENT_HIDDEN_DIM": str(config.get("hidden_dim", 1024)),
        "ORION_STUDENT_OUTPUT_DIM": str(config.get("output_dim", 4096)),
        "ORION_STUDENT_NUM_LAYERS": str(config["num_layers"]),
        "ORION_STUDENT_NUM_HEADS": str(config.get("num_heads", 16)),
        "ORION_STUDENT_DROPOUT": str(config.get("dropout", 0.1)),
        "ORION_STUDENT_WITH_BOUND_LOSS": "True" if bool(config.get("with_bound_loss", True)) else "False",
        "ORION_STUDENT_USE_COL_LOSS": "True" if bool(config.get("use_col_loss", True)) else "False",
    }
    return env


def run_command(cmd, env=None, dry_run=False):
    print("$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def compare_summaries(reference_path, candidate_path):
    reference = _load_json(reference_path)
    candidate = _load_json(candidate_path)
    keys = [
        "driving_score",
        "success_rate",
        "eval_num",
        "ability_mean",
        "ability_overtaking",
        "ability_merging",
        "ability_emergency_brake",
        "ability_give_way",
        "ability_traffic_signs",
        "driving_efficiency",
        "driving_smoothness",
        "crashed_routes",
    ]
    print("\nSummary comparison:")
    print(f"  reference: {reference_path}")
    print(f"  candidate: {candidate_path}")
    for key in keys:
        ref_val = reference.get(key)
        cand_val = candidate.get(key)
        if ref_val is None or cand_val is None:
            delta = "n/a"
        else:
            try:
                delta = cand_val - ref_val
            except TypeError:
                delta = "n/a"
        print(f"  {key}: reference={ref_val} candidate={cand_val} delta={delta}")


def run_fused_closedloop(args, result_dir, config, config_path):
    target = infer_fused_closedloop_target(config)
    output_paths = resolve_eval_output_paths(config, target, args.b2d_root)
    fused_ckpt = os.path.abspath(
        args.fused_ckpt or default_fused_ckpt_path(result_dir=result_dir, config_path=config_path)
    )
    student_env = build_student_env(config)

    if not os.path.isfile(fused_ckpt):
        raise FileNotFoundError(
            f"Fused checkpoint not found: {fused_ckpt}. "
            "Export it first with eval/scripts/export_distill_fused_ckpt.py "
            "or eval/scripts/export_all_distill_fused_last_ckpts.py."
        )

    print(f"fused_ckpt    : {fused_ckpt}")
    print(f"eval_script   : {target['script']}")
    print(f"agent_config  : {target['config']}")
    print(f"algo          : {output_paths['algo']}")
    print(f"planner_type  : {output_paths['planner_type']}")
    print(f"eval_result_dir: {output_paths['result_dir']}")
    print(f"preflight_json: {output_paths['preflight_json']}")
    print(f"summary_json  : {output_paths['summary_json']}")
    print(f"save_path     : {output_paths['save_path']}")
    print("student_env   :")
    for key in sorted(student_env):
        print(f"  {key}={student_env[key]}")

    eval_env = os.environ.copy()
    eval_env.update(student_env)
    eval_env["ORION_LITE_ROOT"] = REPO_ROOT
    eval_env["B2D_ROOT"] = os.path.abspath(args.b2d_root)
    eval_env["CONFIG_PATH"] = target["config"]
    eval_env["CKPT_PATH"] = fused_ckpt
    eval_env["ALGO"] = output_paths["algo"]

    eval_cmd = ["bash", target["script"], args.mode]
    run_command(eval_cmd, env=eval_env, dry_run=args.dry_run)
    return output_paths


def main():
    args = parse_args()

    result_dir = resolve_result_dir(args.result_dir) if args.result_dir else None
    config, config_path, result_dir = load_config(result_dir=result_dir, config_json=args.config_json)
    fused_ckpt = os.path.abspath(args.fused_ckpt or default_fused_ckpt_path(result_dir=result_dir, config_path=config_path))

    print(f"source_run_dir: {result_dir}")
    print(f"config        : {config_path}")
    print(f"mode          : {args.mode}")
    print(f"b2d_root      : {args.b2d_root}")
    print(f"num_layers    : {config['num_layers']}")
    print(f"use_mimic     : {config.get('use_feature_mimic_loss', False)}")

    output_paths = run_fused_closedloop(args, result_dir, config, config_path)

    if args.reference_summary and args.mode in {"postprocess", "all"} and not args.dry_run:
        summary_path = output_paths["summary_json"]
        if os.path.isfile(summary_path):
            compare_summaries(os.path.abspath(args.reference_summary), summary_path)
        else:
            print(f"Candidate summary not found for comparison: {summary_path}")


if __name__ == "__main__":
    main()
