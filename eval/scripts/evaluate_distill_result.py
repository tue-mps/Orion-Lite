#!/usr/bin/env python
"""Prepare and evaluate a distilled Orion-Lite result run."""

import argparse
import json
import os
import subprocess
import sys
from glob import glob


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RESULTS_ROOT = os.path.join(REPO_ROOT, "distill", "results")
DEFAULT_MERGED_ROOT = os.path.join(REPO_ROOT, "eval", "merged_ckpts")
DEFAULT_FUSED_ROOT = os.path.join(REPO_ROOT, "eval", "fused_ckpts")
DEFAULT_B2D_ROOT = "/mnt/adas7tb/jgu/Bench2Drive"
DEFAULT_ORION_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "Orion"))
DEFAULT_LOCAL_ORION_CKPT = os.path.join(REPO_ROOT, "ckpts", "Orion.pth")
DEFAULT_PLANNER_TYPE = "only_traj"
DEFAULT_BASE_CHECKPOINT_ENDPOINT = "eval_bench2drive220"
DEFAULT_BASE_CKPT_CANDIDATES = [
    os.environ.get("ORION_STUDENT_BASE_CKPT"),
    os.path.join(REPO_ROOT, "eval", "base_ckpts", "exp9_stage2_latest.pth"),
    os.path.join(REPO_ROOT, "ckpts", "exp9_stage2_latest.pth"),
    os.path.join(
        DEFAULT_ORION_ROOT,
        "adzoo",
        "orion",
        "work_dirs",
        "exp9_two_stage_20260317_152208",
        "exp9_stage2",
        "latest.pth",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop evaluation wrapper for Orion-Lite distilled runs. "
            "For the public fused-checkpoint workflow, prefer "
            "`evaluate_fused_checkpoint.py`. "
            "This script still exposes the legacy strict-planner backend for "
            "reproducing older results."
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
        "--style",
        choices=["fused_distilled", "strict_planner"],
        default="fused_distilled",
        help=(
            "Evaluation backend. Leave this at `fused_distilled` unless you "
            "explicitly need the legacy OrionStudentPlanner reproduction path. "
            "`strict_planner` is kept as a compatibility/debug fallback."
        ),
    )
    parser.add_argument(
        "--base-ckpt",
        default=None,
        help=(
            "Legacy-only option used by --style strict_planner. "
            "This must be a full OrionStudentPlanner stage-2 checkpoint, not Orion.pth."
        ),
    )
    parser.add_argument(
        "--orion-ckpt",
        default=None,
        help=(
            "Base Orion teacher checkpoint used by --style fused_distilled. "
            f"Defaults to {DEFAULT_LOCAL_ORION_CKPT} when present."
        ),
    )
    parser.add_argument(
        "--orion-root",
        default=DEFAULT_ORION_ROOT,
        help="Original Orion repository root used to resolve fallback distilled checkpoints.",
    )
    parser.add_argument(
        "--distill-ckpt",
        default=None,
        help="Override distilled checkpoint path directly.",
    )
    parser.add_argument(
        "--which",
        choices=["best", "last"],
        default="best",
        help="Which checkpoint under <result-dir>/checkpoints to use when --distill-ckpt is not set.",
    )
    parser.add_argument(
        "--merged-ckpt",
        default=None,
        help="Output path for the merged strict-planner checkpoint.",
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
        "--mode",
        choices=["merge_only", "preflight", "full", "postprocess", "all"],
        default="preflight",
        help="Evaluation stage to run after export.",
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


def resolve_base_ckpt(override):
    if override:
        path = os.path.abspath(override)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Base checkpoint not found: {path}")
        return path

    tried = []
    for candidate in DEFAULT_BASE_CKPT_CANDIDATES:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        tried.append(candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not resolve the base OrionStudentPlanner checkpoint automatically. "
        "Tried: "
        + ", ".join(tried)
        + ". Please pass --base-ckpt."
    )


def resolve_orion_ckpt(override, orion_root):
    candidates = []
    if override:
        candidates.append(os.path.abspath(override))
    if os.path.isfile(DEFAULT_LOCAL_ORION_CKPT):
        candidates.append(os.path.abspath(DEFAULT_LOCAL_ORION_CKPT))
    candidates.append(os.path.join(os.path.abspath(orion_root), "ckpts", "Orion.pth"))

    tried = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        tried.append(candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not resolve Orion.pth automatically. Tried: "
        + ", ".join(tried)
    )


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _matching_score(target_config, candidate_config):
    keys = ("num_layers", "use_feature_mimic_loss", "run_name", "gpu")
    score = 0
    for key in keys:
        if key in target_config and key in candidate_config and target_config[key] == candidate_config[key]:
            score += 1
    return score


def find_orion_run_dir(config, orion_root):
    distill_root = os.path.join(orion_root, "distill")
    config_globs = [
        os.path.join(distill_root, "runs", "*", "config.json"),
        os.path.join(distill_root, "runs", "*", "*", "config.json"),
        os.path.join(distill_root, "runs_ablation_layers", "*", "config.json"),
        os.path.join(distill_root, "runs_ablation_layers", "*", "*", "config.json"),
    ]

    matches = []
    for pattern in config_globs:
        for cfg_path in glob(pattern):
            try:
                candidate = _load_json(cfg_path)
            except Exception:
                continue
            score = _matching_score(config, candidate)
            if score <= 0:
                continue
            if candidate.get("num_layers") != config.get("num_layers"):
                continue
            if bool(candidate.get("use_feature_mimic_loss", False)) != bool(config.get("use_feature_mimic_loss", False)):
                continue
            if config.get("run_name") and candidate.get("run_name") != config.get("run_name"):
                continue
            if "gpu" in config and "gpu" in candidate and candidate.get("gpu") != config.get("gpu"):
                continue
            matches.append((score, os.path.dirname(cfg_path)))

    if not matches:
        return None

    matches.sort(key=lambda item: (-item[0], item[1]))
    return matches[0][1]


def resolve_distill_ckpt(result_dir, override, which="best", config=None, orion_root=DEFAULT_ORION_ROOT):
    if override:
        path = os.path.abspath(override)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Distilled checkpoint not found: {path}")
        return path
    ordered_names = [which, "last" if which == "best" else "best"]
    candidates = []
    if result_dir:
        candidates = [os.path.join(result_dir, "checkpoints", f"{name}.pt") for name in ordered_names]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    fallback_candidates = []
    if config is not None:
        orion_run_dir = find_orion_run_dir(config, orion_root)
        if orion_run_dir:
            fallback_candidates = [
                os.path.join(orion_run_dir, "checkpoints", f"{name}.pt") for name in ordered_names
            ]
            for candidate in fallback_candidates:
                if os.path.isfile(candidate):
                    print(
                        "Local distilled checkpoint missing; using Orion fallback checkpoint: "
                        f"{candidate}"
                    )
                    return candidate

    raise FileNotFoundError(
        "Could not find a distilled checkpoint under result dir. "
        f"Tried local paths: {', '.join(candidates)}"
        + (
            f"; fallback Orion paths: {', '.join(fallback_candidates)}"
            if fallback_candidates
            else ""
        )
    )


def infer_legacy_eval_target(config):
    num_layers = int(config["num_layers"])
    use_feature_mimic_loss = bool(config.get("use_feature_mimic_loss", False))

    scripts_root = os.path.join(REPO_ROOT, "eval", "scripts")
    if num_layers == 6:
        if use_feature_mimic_loss:
            return {
                "family": "6l_with_mimic",
                "script": os.path.join(scripts_root, "run_orion_student_with_mimic_loss_multi_strict.sh"),
                "argv": [],
                "mimic_mode": "with_mimic",
            }
        return {
            "family": "6l_no_mimic",
            "script": os.path.join(scripts_root, "run_orion_student_no_mimic_loss_multi_strict.sh"),
            "argv": [],
            "mimic_mode": "no_mimic",
        }

    if num_layers in {2, 4, 8, 16}:
        mimic_mode = "with_mimic" if use_feature_mimic_loss else "no_mimic"
        return {
            "family": "decoder_ablation",
            "script": os.path.join(scripts_root, "run_orion_student_decoder_ablation_multi_strict.sh"),
            "argv": [str(num_layers), mimic_mode],
            "mimic_mode": mimic_mode,
        }

    raise ValueError(
        f"Unsupported num_layers={num_layers}. Expected 6, 2, 4, 8, or 16."
    )


def infer_fused_closedloop_target(config):
    num_layers = int(config["num_layers"])
    if num_layers == 6:
        return {
            "family": "6l_distilled",
            "script": os.path.join(REPO_ROOT, "eval", "scripts", "run_orion_distilled_fused_multi.sh"),
            "config": os.path.join(REPO_ROOT, "adzoo", "orion", "configs", "orion_distill_new_agent_fused.py"),
            "mimic_mode": "with_mimic" if bool(config.get("use_feature_mimic_loss", False)) else "no_mimic",
        }
    if num_layers in {2, 4, 8, 16}:
        return {
            "family": "decoder_ablation",
            "script": os.path.join(REPO_ROOT, "eval", "scripts", "run_orion_distilled_fused_multi.sh"),
            "config": os.path.join(
                REPO_ROOT,
                "adzoo",
                "orion",
                "configs",
                "ablation_layers",
                "orion_distill_new_agent_ablation_fused.py",
            ),
            "mimic_mode": "with_mimic" if bool(config.get("use_feature_mimic_loss", False)) else "no_mimic",
        }
    raise ValueError(
        f"Unsupported num_layers={num_layers}. Expected 6, 2, 4, 8, or 16."
    )


def infer_algo_name(config, style, target):
    num_layers = int(config["num_layers"])
    use_feature_mimic_loss = bool(config.get("use_feature_mimic_loss", False))

    if style == "strict_planner":
        if target["family"] == "6l_no_mimic":
            return "orion_student_6l_no_mimic_strict"
        if target["family"] == "6l_with_mimic":
            return "orion_student_6l_with_mimic_strict"
        if target["family"] == "decoder_ablation":
            return f"orion_student_decoder{num_layers}l_{target['mimic_mode']}_strict"
        raise ValueError(f"Unsupported target family: {target['family']}")

    if num_layers == 6:
        suffix = "with_mimic" if use_feature_mimic_loss else "no_mimic"
        return f"orion_distilled_6l_{suffix}_fused"
    suffix = "with_mimic" if use_feature_mimic_loss else "no_mimic"
    return f"orion_distilled_decoder{num_layers}l_{suffix}_fused"


def resolve_eval_output_paths(config, style, target, b2d_root, env=None):
    env = env or os.environ
    b2d_root = os.path.abspath(b2d_root)
    algo = infer_algo_name(config, style, target)
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


def default_merged_ckpt_path(result_dir=None, config_path=None):
    stem = artifact_stem_for_paths(result_dir=result_dir, config_path=config_path)
    return os.path.join(DEFAULT_MERGED_ROOT, f"{stem}.pth")


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


def run_strict_planner_flow(args, result_dir, config_path, config, distill_ckpt):
    base_ckpt = resolve_base_ckpt(args.base_ckpt)
    target = infer_legacy_eval_target(config)
    output_paths = resolve_eval_output_paths(config, "strict_planner", target, args.b2d_root)
    merged_ckpt = os.path.abspath(
        args.merged_ckpt or default_merged_ckpt_path(result_dir=result_dir, config_path=config_path)
    )
    os.makedirs(os.path.dirname(merged_ckpt), exist_ok=True)

    print(f"style         : strict_planner")
    print(f"base_ckpt     : {base_ckpt}")
    print(f"merged_ckpt   : {merged_ckpt}")
    print(f"eval_script   : {target['script']}")
    print(f"algo          : {output_paths['algo']}")
    print(f"planner_type  : {output_paths['planner_type']}")
    print(f"eval_result_dir: {output_paths['result_dir']}")
    print(f"preflight_json: {output_paths['preflight_json']}")
    print(f"summary_json  : {output_paths['summary_json']}")
    print(f"save_path     : {output_paths['save_path']}")

    export_cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "eval", "scripts", "export_distill_ckpt.py"),
        "--base-ckpt",
        base_ckpt,
        "--distill-ckpt",
        distill_ckpt,
        "--out-ckpt",
        merged_ckpt,
        "--num-layers",
        str(config["num_layers"]),
    ]
    run_command(export_cmd, dry_run=args.dry_run)

    if args.mode == "merge_only":
        return output_paths

    eval_env = os.environ.copy()
    eval_env["ORION_LITE_ROOT"] = REPO_ROOT
    eval_env["B2D_ROOT"] = os.path.abspath(args.b2d_root)
    eval_env["CKPT_PATH"] = merged_ckpt

    eval_cmd = ["bash", target["script"], *target["argv"], args.mode]
    run_command(eval_cmd, env=eval_env, dry_run=args.dry_run)
    return output_paths


def run_fused_distilled_flow(args, result_dir, config, distill_ckpt):
    orion_ckpt = resolve_orion_ckpt(args.orion_ckpt, args.orion_root)
    target = infer_fused_closedloop_target(config)
    output_paths = resolve_eval_output_paths(config, "fused_distilled", target, args.b2d_root)
    fused_ckpt = os.path.abspath(args.fused_ckpt or default_fused_ckpt_path(result_dir))
    os.makedirs(os.path.dirname(fused_ckpt), exist_ok=True)

    student_env = build_student_env(config)
    reuse_existing_fused = os.path.isfile(fused_ckpt) and not args.rebuild_fused

    print(f"style         : fused_distilled")
    print(f"orion_ckpt    : {orion_ckpt}")
    print(f"fused_ckpt    : {fused_ckpt}")
    print(f"reuse_fused   : {reuse_existing_fused}")
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
        run_command(export_cmd, dry_run=args.dry_run)

    if args.mode == "merge_only":
        return output_paths

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
    distill_ckpt = None
    fused_ckpt = os.path.abspath(args.fused_ckpt or default_fused_ckpt_path(result_dir=result_dir, config_path=config_path))
    if args.style != "fused_distilled" or args.rebuild_fused or not os.path.isfile(fused_ckpt):
        distill_ckpt = resolve_distill_ckpt(
            result_dir,
            args.distill_ckpt,
            which=args.which,
            config=config,
            orion_root=os.path.abspath(args.orion_root),
        )

    print(f"source_run_dir: {result_dir}")
    print(f"config        : {config_path}")
    print(f"distill_ckpt  : {distill_ckpt}")
    print(f"which         : {args.which}")
    print(f"style         : {args.style}")
    print(f"mode          : {args.mode}")
    print(f"b2d_root      : {args.b2d_root}")
    print(f"orion_root    : {os.path.abspath(args.orion_root)}")
    print(f"num_layers    : {config['num_layers']}")
    print(f"use_mimic     : {config.get('use_feature_mimic_loss', False)}")
    print(f"fused_ckpt    : {fused_ckpt}")

    if args.style == "strict_planner":
        if distill_ckpt is None:
            raise ValueError("strict_planner style requires a distilled checkpoint; pass --rebuild-fused only for fused_distilled.")
        output_paths = run_strict_planner_flow(args, result_dir, config_path, config, distill_ckpt)
    else:
        output_paths = run_fused_distilled_flow(args, result_dir, config, distill_ckpt)

    if args.reference_summary and args.mode in {"postprocess", "all"} and not args.dry_run:
        summary_path = output_paths["summary_json"]
        if os.path.isfile(summary_path):
            compare_summaries(os.path.abspath(args.reference_summary), summary_path)
        else:
            print(f"Candidate summary not found for comparison: {summary_path}")


if __name__ == "__main__":
    main()
