#!/usr/bin/env python
"""Compare reference distill data against Orion-Lite test collection output."""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np


DEFAULT_REFERENCE_DIR = "/mnt/adas7tb/jgu/Orion/distill/distill_data"
DEFAULT_CANDIDATE_DIR = "/mnt/adas7tb/jgu/Orion-Lite/distill_data_test"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare reference distillation data against a test distill dataset."
    )
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument(
        "--match",
        choices=["auto", "name", "index"],
        default="auto",
        help="How to pair files for numeric comparison.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="How many files per split to inspect for structure summary.",
    )
    parser.add_argument(
        "--pair-limit",
        type=int,
        default=20,
        help="How many paired files per split to compare numerically.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the structured report as JSON.",
    )
    return parser.parse_args()


def normalize_name(path):
    name = os.path.basename(path)
    if name.endswith(".npy.npz"):
        return name[:-8] + ".npz"
    return name


def format_shape(shape):
    return "(" + ", ".join(str(x) for x in shape) + ")"


def sorted_npz_files(root):
    if not os.path.isdir(root):
        return []
    files = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if os.path.isfile(path) and entry.endswith(".npz"):
            files.append(path)
    return files


def summarize_file_set(files, sample_limit):
    summary = {
        "count": len(files),
        "sample_files": [os.path.basename(p) for p in files[: min(5, len(files))]],
        "keys": {},
    }
    if not files:
        return summary

    for path in files[:sample_limit]:
        with np.load(path) as data:
            present_keys = set(data.files)
            for key in present_keys:
                arr = data[key]
                key_summary = summary["keys"].setdefault(
                    key,
                    {
                        "present_count": 0,
                        "dtypes": Counter(),
                        "shapes": Counter(),
                    },
                )
                key_summary["present_count"] += 1
                key_summary["dtypes"][str(arr.dtype)] += 1
                key_summary["shapes"][format_shape(arr.shape)] += 1

    for key, key_summary in summary["keys"].items():
        key_summary["dtypes"] = dict(key_summary["dtypes"].most_common())
        key_summary["shapes"] = dict(key_summary["shapes"].most_common())

    return summary


def choose_pairs(reference_files, candidate_files, match_mode, pair_limit):
    if match_mode in ("auto", "name"):
        ref_by_name = {normalize_name(path): path for path in reference_files}
        cand_by_name = {normalize_name(path): path for path in candidate_files}
        common_names = sorted(set(ref_by_name) & set(cand_by_name))
        if common_names:
            pairs = [(name, ref_by_name[name], cand_by_name[name]) for name in common_names[:pair_limit]]
            return "name", pairs
        if match_mode == "name":
            return "name", []

    pairs = []
    for ref_path, cand_path in zip(reference_files[:pair_limit], candidate_files[:pair_limit]):
        pairs.append((f"{os.path.basename(ref_path)} <-> {os.path.basename(cand_path)}", ref_path, cand_path))
    return "index", pairs


def compare_pair(reference_path, candidate_path):
    result = {
        "reference_file": os.path.basename(reference_path),
        "candidate_file": os.path.basename(candidate_path),
        "reference_norm_name": normalize_name(reference_path),
        "candidate_norm_name": normalize_name(candidate_path),
        "reference_only_keys": [],
        "candidate_only_keys": [],
        "per_key": {},
    }

    with np.load(reference_path) as ref_data, np.load(candidate_path) as cand_data:
        ref_keys = set(ref_data.files)
        cand_keys = set(cand_data.files)
        result["reference_only_keys"] = sorted(ref_keys - cand_keys)
        result["candidate_only_keys"] = sorted(cand_keys - ref_keys)

        for key in sorted(ref_keys & cand_keys):
            ref_arr = ref_data[key]
            cand_arr = cand_data[key]
            entry = {
                "reference_shape": list(ref_arr.shape),
                "candidate_shape": list(cand_arr.shape),
                "reference_dtype": str(ref_arr.dtype),
                "candidate_dtype": str(cand_arr.dtype),
                "shape_match": tuple(ref_arr.shape) == tuple(cand_arr.shape),
                "dtype_match": str(ref_arr.dtype) == str(cand_arr.dtype),
            }

            if entry["shape_match"]:
                ref_f = ref_arr.astype(np.float32, copy=False)
                cand_f = cand_arr.astype(np.float32, copy=False)
                abs_diff = np.abs(ref_f - cand_f)
                entry["mean_abs_diff"] = float(abs_diff.mean())
                entry["max_abs_diff"] = float(abs_diff.max())
                entry["exact_equal"] = bool(np.array_equal(ref_arr, cand_arr))
            result["per_key"][key] = entry

    return result


def summarize_pairs(pair_reports):
    summary = {
        "pair_count": len(pair_reports),
        "reference_only_keys": Counter(),
        "candidate_only_keys": Counter(),
        "per_key": defaultdict(lambda: {
            "count": 0,
            "shape_match_count": 0,
            "dtype_match_count": 0,
            "exact_equal_count": 0,
            "mean_abs_diff_sum": 0.0,
            "max_abs_diff_max": 0.0,
            "mean_abs_diff_count": 0,
        }),
    }

    for report in pair_reports:
        summary["reference_only_keys"].update(report["reference_only_keys"])
        summary["candidate_only_keys"].update(report["candidate_only_keys"])
        for key, entry in report["per_key"].items():
            key_summary = summary["per_key"][key]
            key_summary["count"] += 1
            if entry["shape_match"]:
                key_summary["shape_match_count"] += 1
            if entry["dtype_match"]:
                key_summary["dtype_match_count"] += 1
            if entry.get("exact_equal", False):
                key_summary["exact_equal_count"] += 1
            if "mean_abs_diff" in entry:
                key_summary["mean_abs_diff_sum"] += entry["mean_abs_diff"]
                key_summary["mean_abs_diff_count"] += 1
                key_summary["max_abs_diff_max"] = max(
                    key_summary["max_abs_diff_max"], entry["max_abs_diff"]
                )

    output = {
        "pair_count": summary["pair_count"],
        "reference_only_keys": dict(summary["reference_only_keys"].most_common()),
        "candidate_only_keys": dict(summary["candidate_only_keys"].most_common()),
        "per_key": {},
    }

    for key in sorted(summary["per_key"]):
        key_summary = summary["per_key"][key]
        mean_abs_diff = None
        if key_summary["mean_abs_diff_count"] > 0:
            mean_abs_diff = key_summary["mean_abs_diff_sum"] / key_summary["mean_abs_diff_count"]
        output["per_key"][key] = {
            "count": key_summary["count"],
            "shape_match_count": key_summary["shape_match_count"],
            "dtype_match_count": key_summary["dtype_match_count"],
            "exact_equal_count": key_summary["exact_equal_count"],
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": key_summary["max_abs_diff_max"] if key_summary["mean_abs_diff_count"] > 0 else None,
        }
    return output


def print_file_summary(label, summary, inspected_count):
    print(f"{label}: {summary['count']} files")
    if summary["sample_files"]:
        print(f"  sample files: {', '.join(summary['sample_files'])}")
    else:
        print("  sample files: none")
    print(f"  inspected for structure: {min(summary['count'], inspected_count)}")
    if not summary["keys"]:
        return
    for key in sorted(summary["keys"]):
        key_summary = summary["keys"][key]
        shapes = ", ".join(
            f"{shape} x{count}" for shape, count in key_summary["shapes"].items()
        )
        dtypes = ", ".join(
            f"{dtype} x{count}" for dtype, count in key_summary["dtypes"].items()
        )
        print(f"  {key}: present {key_summary['present_count']} times")
        print(f"    dtypes: {dtypes}")
        print(f"    shapes: {shapes}")


def print_pair_summary(match_kind, pair_summary):
    print(f"paired comparison mode: {match_kind}")
    print(f"paired files compared: {pair_summary['pair_count']}")
    if pair_summary["reference_only_keys"]:
        print(f"reference-only key counts: {pair_summary['reference_only_keys']}")
    if pair_summary["candidate_only_keys"]:
        print(f"candidate-only key counts: {pair_summary['candidate_only_keys']}")
    for key in sorted(pair_summary["per_key"]):
        key_summary = pair_summary["per_key"][key]
        mean_abs = key_summary["mean_abs_diff"]
        mean_abs_text = "n/a" if mean_abs is None else f"{mean_abs:.6g}"
        max_abs = key_summary["max_abs_diff"]
        max_abs_text = "n/a" if max_abs is None else f"{max_abs:.6g}"
        print(
            f"  {key}: shape {key_summary['shape_match_count']}/{key_summary['count']}, "
            f"dtype {key_summary['dtype_match_count']}/{key_summary['count']}, "
            f"exact {key_summary['exact_equal_count']}/{key_summary['count']}, "
            f"mean_abs_diff {mean_abs_text}, max_abs_diff {max_abs_text}"
        )


def compare_split(reference_root, candidate_root, split, sample_limit, pair_limit, match_mode):
    reference_files = sorted_npz_files(os.path.join(reference_root, split))
    candidate_files = sorted_npz_files(os.path.join(candidate_root, split))

    reference_summary = summarize_file_set(reference_files, sample_limit)
    candidate_summary = summarize_file_set(candidate_files, sample_limit)
    match_kind, pairs = choose_pairs(reference_files, candidate_files, match_mode, pair_limit)
    pair_reports = [compare_pair(ref_path, cand_path) for _, ref_path, cand_path in pairs]
    pair_summary = summarize_pairs(pair_reports)

    return {
        "split": split,
        "reference": reference_summary,
        "candidate": candidate_summary,
        "pair_mode": match_kind,
        "pairs": pair_reports,
        "pair_summary": pair_summary,
    }


def main():
    args = parse_args()
    splits = ["train", "val"] if args.split == "all" else [args.split]

    report = {
        "reference_dir": args.reference_dir,
        "candidate_dir": args.candidate_dir,
        "split": args.split,
        "match": args.match,
        "sample_limit": args.sample_limit,
        "pair_limit": args.pair_limit,
        "results": [],
    }

    for split in splits:
        result = compare_split(
            args.reference_dir,
            args.candidate_dir,
            split,
            args.sample_limit,
            args.pair_limit,
            args.match,
        )
        report["results"].append(result)

        print("=" * 72)
        print(f"Split: {split}")
        print_file_summary("reference", result["reference"], args.sample_limit)
        print_file_summary("candidate", result["candidate"], args.sample_limit)
        print_pair_summary(result["pair_mode"], result["pair_summary"])

    if args.output_json:
        serializable = json.loads(json.dumps(report))
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, sort_keys=True)
        print("=" * 72)
        print(f"Wrote JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
