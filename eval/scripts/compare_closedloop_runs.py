#!/usr/bin/env python3
"""Compare two closed-loop Bench2Drive result sets and emit a mismatch subset."""

import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        required=True,
        help="Current result dir or json file (e.g. split result dir).",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference result dir or json file (e.g. old merged.json).",
    )
    parser.add_argument(
        "--base-routes",
        default="/mnt/adas7tb/jgu/Bench2Drive/leaderboard/data/bench2drive220.xml",
        help="Base Bench2Drive route xml used to generate the mismatch subset xml.",
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/adas7tb/jgu/Orion-Lite/eval/route_diffs/orion_distilled_6l_with_mimic_fused_vs_eval_mimic",
        help="Directory to write comparison artifacts.",
    )
    parser.add_argument(
        "--score-tol",
        type=float,
        default=1e-6,
        help="Tolerance for score comparisons.",
    )
    return parser.parse_args()


def extract_records(data):
    if not isinstance(data, dict):
        raise ValueError("Expected a dict-like json object")
    return data.get("_checkpoint", {}).get("records") or data.get("records") or []


def load_records(path):
    json_paths = []
    if os.path.isdir(path):
        split_paths = sorted(glob.glob(os.path.join(path, "eval_bench2drive220_*.json")))
        if split_paths:
            json_paths = split_paths
        else:
            for candidate in ("merged.json", "eval_mimic.json", "summary.json"):
                candidate_path = os.path.join(path, candidate)
                if os.path.isfile(candidate_path):
                    json_paths = [candidate_path]
                    break
    else:
        json_paths = [path]

    if not json_paths:
        raise FileNotFoundError(f"No result jsons found under: {path}")

    records = []
    sources = []
    for json_path in json_paths:
        with open(json_path) as f:
            data = json.load(f)
        part_records = extract_records(data)
        records.extend(part_records)
        sources.append(json_path)
    return records, sources


def score_triplet(record):
    scores = record.get("scores", {})
    return {
        "score_route": float(scores.get("score_route", 0.0)),
        "score_penalty": float(scores.get("score_penalty", 0.0)),
        "score_composed": float(scores.get("score_composed", 0.0)),
    }


def summarize_infractions(record):
    infractions = record.get("infractions", {})
    summary = OrderedDict()
    for key, value in infractions.items():
        count = len(value) if isinstance(value, list) else 0
        if count:
            summary[key] = count
    return summary


def route_number_from_route_id(route_id):
    match = re.match(r"RouteScenario_(\d+)_rep\d+$", route_id)
    if match:
        return match.group(1)
    if route_id.isdigit():
        return route_id
    raise ValueError(f"Unsupported route id format: {route_id}")


def is_success(record):
    if record.get("status") not in ("Completed", "Perfect"):
        return False
    infractions = record.get("infractions", {})
    for key, value in infractions.items():
        if key == "min_speed_infractions":
            continue
        if isinstance(value, list) and len(value) > 0:
            return False
    return True


def summarize_subset(records):
    count = len(records)
    driving_score = sum(score_triplet(record)["score_composed"] for record in records) / count if count else 0.0
    success_rate = 100.0 * sum(1 for record in records if is_success(record)) / count if count else 0.0
    completed_rate = 100.0 * sum(1 for record in records if record.get("status") == "Completed") / count if count else 0.0
    return OrderedDict(
        count=count,
        driving_score=driving_score,
        success_rate=success_rate,
        completed_rate=completed_rate,
    )


def build_subset_xml(base_routes_path, route_numbers, out_path):
    tree = ET.parse(base_routes_path)
    root = tree.getroot()
    route_numbers = set(route_numbers)

    new_root = ET.Element("routes")
    found = []
    for route in root.findall("route"):
        route_id = route.attrib.get("id")
        if route_id in route_numbers:
            new_root.append(route)
            found.append(route_id)

    missing = sorted(route_numbers - set(found), key=int)
    if missing:
        raise ValueError(f"Missing routes in base xml: {missing}")

    new_tree = ET.ElementTree(new_root)
    new_tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    current_records, current_sources = load_records(args.current)
    reference_records, reference_sources = load_records(args.reference)

    current_by_id = OrderedDict((record["route_id"], record) for record in current_records)
    reference_by_id = OrderedDict((record["route_id"], record) for record in reference_records)

    common_ids = sorted(set(current_by_id) & set(reference_by_id), key=lambda x: int(route_number_from_route_id(x)))
    current_only = sorted(set(current_by_id) - set(reference_by_id), key=lambda x: int(route_number_from_route_id(x)))
    reference_only = sorted(set(reference_by_id) - set(current_by_id), key=lambda x: int(route_number_from_route_id(x)))

    mismatches = []
    for route_id in common_ids:
        current_record = current_by_id[route_id]
        reference_record = reference_by_id[route_id]
        current_scores = score_triplet(current_record)
        reference_scores = score_triplet(reference_record)
        same_status = current_record.get("status") == reference_record.get("status")
        same_scores = all(
            abs(current_scores[key] - reference_scores[key]) <= args.score_tol
            for key in current_scores
        )
        if same_status and same_scores:
            continue

        mismatches.append(
            OrderedDict(
                route_id=route_id,
                route_number=route_number_from_route_id(route_id),
                current_status=current_record.get("status"),
                reference_status=reference_record.get("status"),
                current_scores=current_scores,
                reference_scores=reference_scores,
                current_num_infractions=current_record.get("num_infractions"),
                reference_num_infractions=reference_record.get("num_infractions"),
                current_infractions=summarize_infractions(current_record),
                reference_infractions=summarize_infractions(reference_record),
            )
        )

    mismatch_route_ids = [item["route_id"] for item in mismatches]
    mismatch_route_numbers = [item["route_number"] for item in mismatches]
    routes_subset = ",".join(mismatch_route_numbers)
    current_common_records = [current_by_id[route_id] for route_id in common_ids]
    reference_common_records = [reference_by_id[route_id] for route_id in common_ids]
    current_summary = summarize_subset(current_common_records)
    reference_summary = summarize_subset(reference_common_records)

    report = OrderedDict(
        current_sources=current_sources,
        reference_sources=reference_sources,
        current_record_count=len(current_records),
        reference_record_count=len(reference_records),
        common_route_count=len(common_ids),
        mismatch_count=len(mismatches),
        current_only=current_only,
        reference_only=reference_only,
        mismatches=mismatches,
        mismatch_route_ids=mismatch_route_ids,
        mismatch_route_numbers=mismatch_route_numbers,
        routes_subset=routes_subset,
        current_common_summary=current_summary,
        reference_common_summary=reference_summary,
        delta_summary=OrderedDict(
            driving_score=current_summary["driving_score"] - reference_summary["driving_score"],
            success_rate=current_summary["success_rate"] - reference_summary["success_rate"],
            completed_rate=current_summary["completed_rate"] - reference_summary["completed_rate"],
        ),
    )

    report_path = os.path.join(args.out_dir, "mismatch_report.json")
    route_ids_path = os.path.join(args.out_dir, "mismatch_route_ids.txt")
    route_numbers_path = os.path.join(args.out_dir, "mismatch_route_numbers.txt")
    subset_xml_path = os.path.join(args.out_dir, "mismatch_routes.xml")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(route_ids_path, "w") as f:
        f.write("\n".join(mismatch_route_ids) + ("\n" if mismatch_route_ids else ""))
    with open(route_numbers_path, "w") as f:
        f.write("\n".join(mismatch_route_numbers) + ("\n" if mismatch_route_numbers else ""))

    if mismatch_route_numbers:
        build_subset_xml(args.base_routes, mismatch_route_numbers, subset_xml_path)

    print(f"current_sources   : {current_sources}")
    print(f"reference_sources : {reference_sources}")
    print(f"current_records   : {len(current_records)}")
    print(f"reference_records : {len(reference_records)}")
    print(f"common_routes     : {len(common_ids)}")
    print(f"mismatch_count    : {len(mismatches)}")
    print(f"current_summary   : {json.dumps(current_summary)}")
    print(f"reference_summary : {json.dumps(reference_summary)}")
    if current_only:
        print(f"current_only      : {len(current_only)}")
    if reference_only:
        print(f"reference_only    : {len(reference_only)}")
    print(f"report_json       : {report_path}")
    print(f"route_ids_txt     : {route_ids_path}")
    print(f"route_numbers_txt : {route_numbers_path}")
    if mismatch_route_numbers:
        print(f"subset_xml        : {subset_xml_path}")
        print("mismatch_routes   :")
        for item in mismatches:
            print(
                f"  {item['route_id']}: "
                f"current={item['current_status']} {item['current_scores']} | "
                f"reference={item['reference_status']} {item['reference_scores']}"
            )
        print("routes_subset_env :")
        print(f"  {routes_subset}")


if __name__ == "__main__":
    main()
