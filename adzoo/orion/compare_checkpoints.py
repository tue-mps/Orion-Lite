#!/usr/bin/env python3
import argparse
import collections
import hashlib
import os
import gc
from typing import Dict, Tuple

import torch


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def summarize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict:
    bytes_by_prefix = collections.Counter()
    dtype_counter = collections.Counter()

    tensor_count = 0
    total_bytes = 0
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        tensor_count += 1
        t_bytes = tensor_nbytes(value)
        total_bytes += t_bytes
        bytes_by_prefix[key.split(".", 1)[0]] += t_bytes
        dtype_counter[str(value.dtype)] += 1

    return {
        "key_count": len(state_dict),
        "tensor_count": tensor_count,
        "total_bytes": total_bytes,
        "bytes_by_prefix": bytes_by_prefix,
        "dtype_counter": dtype_counter,
    }


def summarize_optimizer(optimizer_obj) -> Dict:
    summary = {
        "present": False,
        "param_group_count": 0,
        "state_entry_count": 0,
        "tensor_count": 0,
        "total_bytes": 0,
        "dtype_counter": collections.Counter(),
        "tensor_fields": collections.Counter(),
        "tensor_field_bytes": collections.Counter(),
    }
    if not isinstance(optimizer_obj, dict):
        return summary

    summary["present"] = True
    state = optimizer_obj.get("state", {})
    param_groups = optimizer_obj.get("param_groups", [])
    summary["param_group_count"] = len(param_groups)
    summary["state_entry_count"] = len(state) if isinstance(state, dict) else 0

    if not isinstance(state, dict):
        return summary

    for per_param_state in state.values():
        if not isinstance(per_param_state, dict):
            continue
        for field_name, value in per_param_state.items():
            if not torch.is_tensor(value):
                continue
            t_bytes = tensor_nbytes(value)
            summary["tensor_count"] += 1
            summary["total_bytes"] += t_bytes
            summary["dtype_counter"][str(value.dtype)] += 1
            summary["tensor_fields"][field_name] += 1
            summary["tensor_field_bytes"][field_name] += t_bytes
    return summary


def compare_state_dicts(
    state_dict_a: Dict[str, torch.Tensor], state_dict_b: Dict[str, torch.Tensor]
) -> Dict:
    keys_a = set(state_dict_a.keys())
    keys_b = set(state_dict_b.keys())
    common = keys_a & keys_b

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    shape_mismatch = []
    dtype_mismatch = []
    for key in sorted(common):
        va = state_dict_a[key]
        vb = state_dict_b[key]
        if not (torch.is_tensor(va) and torch.is_tensor(vb)):
            continue
        if tuple(va.shape) != tuple(vb.shape):
            shape_mismatch.append((key, tuple(va.shape), tuple(vb.shape)))
        if va.dtype != vb.dtype:
            dtype_mismatch.append((key, str(va.dtype), str(vb.dtype)))

    return {
        "keys_a": len(keys_a),
        "keys_b": len(keys_b),
        "common_keys": len(common),
        "only_a": only_a,
        "only_b": only_b,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
    }


def load_checkpoint(path: str):
    return torch.load(path, map_location="meta", weights_only=False)


def file_digest(path: str, algo: str = "md5", chunk_size: int = 16 * 1024 * 1024) -> str:
    hasher = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def tensor_digest(tensor: torch.Tensor, algo: str = "md5") -> str:
    hasher = hashlib.new(algo)
    hasher.update(str(tensor.dtype).encode("utf-8"))
    hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
    arr = tensor.detach().cpu().contiguous().numpy()
    hasher.update(memoryview(arr).cast("B"))
    return hasher.hexdigest()


def state_dict_value_digests(
    checkpoint_path: str, algo: str = "md5", progress_every: int = 200
) -> Dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", {})
    per_key = {}
    global_hasher = hashlib.new(algo)
    tensor_count = 0
    non_tensor_count = 0

    sorted_keys = sorted(state_dict.keys())
    for idx, key in enumerate(sorted_keys, start=1):
        value = state_dict[key]
        if not torch.is_tensor(value):
            non_tensor_count += 1
            continue
        digest = tensor_digest(value, algo=algo)
        per_key[key] = digest
        global_hasher.update(key.encode("utf-8"))
        global_hasher.update(digest.encode("utf-8"))
        tensor_count += 1
        if progress_every > 0 and idx % progress_every == 0:
            print(
                f"[value-hash] {os.path.basename(checkpoint_path)}: "
                f"{idx}/{len(sorted_keys)} keys processed"
            )

    # Release memory aggressively before loading the next checkpoint.
    del ckpt
    gc.collect()

    return {
        "per_key": per_key,
        "global_digest": global_hasher.hexdigest(),
        "tensor_count": tensor_count,
        "non_tensor_count": non_tensor_count,
    }


def print_ckpt_summary(name: str, path: str, ckpt: Dict, top_n_prefixes: int) -> Tuple[Dict, Dict]:
    file_size = os.path.getsize(path)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    state_dict = ckpt.get("state_dict", {}) if isinstance(ckpt, dict) else {}
    optimizer_obj = ckpt.get("optimizer") if isinstance(ckpt, dict) else None

    state_summary = summarize_state_dict(state_dict)
    opt_summary = summarize_optimizer(optimizer_obj)

    print(f"\n=== {name} ===")
    print(f"path: {path}")
    print(f"file_size: {file_size} ({human_bytes(file_size)})")
    print(f"meta: epoch={meta.get('epoch')}, iter={meta.get('iter')}, exp_name={meta.get('exp_name')}")
    cfg_text = meta.get("config", "")
    if isinstance(cfg_text, str):
        print(f"meta.config has FreezeVisionEncoderHook: {'FreezeVisionEncoderHook' in cfg_text}")

    print("\n[model state_dict]")
    print(
        f"keys={state_summary['key_count']}, tensors={state_summary['tensor_count']}, "
        f"bytes={state_summary['total_bytes']} ({human_bytes(state_summary['total_bytes'])})"
    )
    print(f"dtypes={dict(state_summary['dtype_counter'])}")

    sorted_prefixes = sorted(
        state_summary["bytes_by_prefix"].items(), key=lambda x: x[1], reverse=True
    )
    print(f"top {top_n_prefixes} module prefixes by bytes:")
    for prefix, num_bytes in sorted_prefixes[:top_n_prefixes]:
        print(f"  - {prefix}: {num_bytes} ({human_bytes(num_bytes)})")

    print("\n[optimizer]")
    if not opt_summary["present"]:
        print("optimizer not found in checkpoint")
    else:
        print(
            f"param_groups={opt_summary['param_group_count']}, "
            f"state_entries={opt_summary['state_entry_count']}, "
            f"tensor_count={opt_summary['tensor_count']}, "
            f"bytes={opt_summary['total_bytes']} ({human_bytes(opt_summary['total_bytes'])})"
        )
        print(f"optimizer tensor dtypes={dict(opt_summary['dtype_counter'])}")
        sorted_fields = sorted(
            opt_summary["tensor_field_bytes"].items(), key=lambda x: x[1], reverse=True
        )
        print("optimizer tensor fields by bytes:")
        for field, num_bytes in sorted_fields:
            count = opt_summary["tensor_fields"][field]
            print(f"  - {field}: count={count}, bytes={num_bytes} ({human_bytes(num_bytes)})")
    return state_summary, opt_summary


def main():
    parser = argparse.ArgumentParser(
        description="Compare two PyTorch/MMCV checkpoints (model + optimizer summaries)."
    )
    parser.add_argument("checkpoint_a", help="Path to first checkpoint.")
    parser.add_argument("checkpoint_b", help="Path to second checkpoint.")
    parser.add_argument(
        "--name-a", default="A", help="Display name for checkpoint A (default: A)."
    )
    parser.add_argument(
        "--name-b", default="B", help="Display name for checkpoint B (default: B)."
    )
    parser.add_argument(
        "--top-n-prefixes",
        type=int,
        default=12,
        help="How many top module prefixes to print for model state_dict bytes.",
    )
    parser.add_argument(
        "--show-key-diff-limit",
        type=int,
        default=20,
        help="How many unmatched state_dict keys to print for each side.",
    )
    parser.add_argument(
        "--compare-values",
        action="store_true",
        help="Compare actual state_dict tensor values using hashes (expensive).",
    )
    parser.add_argument(
        "--hash-algo",
        choices=["md5", "sha256"],
        default="md5",
        help="Hash algorithm for --compare-values and file digest (default: md5).",
    )
    args = parser.parse_args()

    ckpt_a = load_checkpoint(args.checkpoint_a)
    ckpt_b = load_checkpoint(args.checkpoint_b)

    state_a, opt_a = print_ckpt_summary(
        args.name_a, args.checkpoint_a, ckpt_a, args.top_n_prefixes
    )
    state_b, opt_b = print_ckpt_summary(
        args.name_b, args.checkpoint_b, ckpt_b, args.top_n_prefixes
    )

    file_size_a = os.path.getsize(args.checkpoint_a)
    file_size_b = os.path.getsize(args.checkpoint_b)
    comp = compare_state_dicts(ckpt_a.get("state_dict", {}), ckpt_b.get("state_dict", {}))

    print("\n=== Comparison Summary ===")
    print(
        f"file_size_delta(A-B): {file_size_a - file_size_b} "
        f"({human_bytes(file_size_a - file_size_b)})"
    )
    print(
        f"model_state_bytes_delta(A-B): {state_a['total_bytes'] - state_b['total_bytes']} "
        f"({human_bytes(state_a['total_bytes'] - state_b['total_bytes'])})"
    )
    print(
        f"optimizer_bytes_delta(A-B): {opt_a['total_bytes'] - opt_b['total_bytes']} "
        f"({human_bytes(opt_a['total_bytes'] - opt_b['total_bytes'])})"
    )
    print(
        f"optimizer_state_entries_delta(A-B): "
        f"{opt_a['state_entry_count'] - opt_b['state_entry_count']}"
    )
    print(
        f"state_dict keys: A={comp['keys_a']}, B={comp['keys_b']}, common={comp['common_keys']}, "
        f"only_A={len(comp['only_a'])}, only_B={len(comp['only_b'])}"
    )
    print(
        f"shape_mismatch={len(comp['shape_mismatch'])}, dtype_mismatch={len(comp['dtype_mismatch'])}"
    )

    if comp["only_a"]:
        print(f"first {args.show_key_diff_limit} keys only in A:")
        for key in comp["only_a"][: args.show_key_diff_limit]:
            print(f"  - {key}")
    if comp["only_b"]:
        print(f"first {args.show_key_diff_limit} keys only in B:")
        for key in comp["only_b"][: args.show_key_diff_limit]:
            print(f"  - {key}")
    if comp["shape_mismatch"]:
        print(f"first {args.show_key_diff_limit} shape mismatches:")
        for key, shape_a, shape_b in comp["shape_mismatch"][: args.show_key_diff_limit]:
            print(f"  - {key}: A{shape_a} vs B{shape_b}")
    if comp["dtype_mismatch"]:
        print(f"first {args.show_key_diff_limit} dtype mismatches:")
        for key, dtype_a, dtype_b in comp["dtype_mismatch"][: args.show_key_diff_limit]:
            print(f"  - {key}: A({dtype_a}) vs B({dtype_b})")

    model_delta = abs(state_a["total_bytes"] - state_b["total_bytes"])
    optimizer_delta = abs(opt_a["total_bytes"] - opt_b["total_bytes"])
    file_delta = abs(file_size_a - file_size_b)
    if model_delta == 0 and optimizer_delta > 0:
        print(
            "\nLikely cause: model weights are effectively the same size, "
            "and the checkpoint size difference comes mainly from optimizer state."
        )
    elif optimizer_delta > model_delta and optimizer_delta > file_delta * 0.5:
        print(
            "\nLikely cause: most of the size difference is from optimizer state "
            "(often due to different sets of trainable parameters)."
        )

    if args.compare_values:
        print("\n=== Value Hash Comparison ===")
        print(
            f"Computing file {args.hash_algo} and state_dict value hashes. "
            "This is CPU- and I/O-heavy."
        )
        file_hash_a = file_digest(args.checkpoint_a, algo=args.hash_algo)
        file_hash_b = file_digest(args.checkpoint_b, algo=args.hash_algo)
        print(f"file_{args.hash_algo}({args.name_a}) = {file_hash_a}")
        print(f"file_{args.hash_algo}({args.name_b}) = {file_hash_b}")
        print(f"whole file hash equal: {file_hash_a == file_hash_b}")

        value_hash_a = state_dict_value_digests(args.checkpoint_a, algo=args.hash_algo)
        value_hash_b = state_dict_value_digests(args.checkpoint_b, algo=args.hash_algo)
        print(f"state_dict_value_{args.hash_algo}({args.name_a}) = {value_hash_a['global_digest']}")
        print(f"state_dict_value_{args.hash_algo}({args.name_b}) = {value_hash_b['global_digest']}")
        print(
            f"state_dict value hash equal: "
            f"{value_hash_a['global_digest'] == value_hash_b['global_digest']}"
        )

        keys_a = set(value_hash_a["per_key"].keys())
        keys_b = set(value_hash_b["per_key"].keys())
        common = sorted(keys_a & keys_b)
        changed = [
            key
            for key in common
            if value_hash_a["per_key"][key] != value_hash_b["per_key"][key]
        ]
        only_a_hash = sorted(keys_a - keys_b)
        only_b_hash = sorted(keys_b - keys_a)
        print(
            f"hashed tensor keys: A={len(keys_a)}, B={len(keys_b)}, "
            f"common={len(common)}, changed_values={len(changed)}"
        )
        if only_a_hash:
            print(f"first {args.show_key_diff_limit} hashed keys only in A:")
            for key in only_a_hash[: args.show_key_diff_limit]:
                print(f"  - {key}")
        if only_b_hash:
            print(f"first {args.show_key_diff_limit} hashed keys only in B:")
            for key in only_b_hash[: args.show_key_diff_limit]:
                print(f"  - {key}")
        if changed:
            print(f"first {args.show_key_diff_limit} keys with different tensor values:")
            for key in changed[: args.show_key_diff_limit]:
                print(
                    f"  - {key}: "
                    f"{args.name_a}={value_hash_a['per_key'][key]} "
                    f"{args.name_b}={value_hash_b['per_key'][key]}"
                )


if __name__ == "__main__":
    main()
