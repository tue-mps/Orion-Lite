#!/usr/bin/env python3
"""
Count trainable (saved) parameters in a PyTorch checkpoint (.pth).
Usage:
    python count_checkpoint_params.py /path/to/checkpoint.pth
"""
import argparse
import sys

import torch


def count_params(checkpoint_path: str) -> dict:
    """Load checkpoint and count parameters from state_dict."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # mmcv/training checkpoints use 'state_dict'; raw state_dict is the dict itself
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    total_params = 0
    tensor_count = 0
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            total_params += value.numel()
            tensor_count += 1

    return {
        "total_params": total_params,
        "tensor_count": tensor_count,
        "total_params_M": total_params / 1e6,
    }


def main():
    parser = argparse.ArgumentParser(description="Count parameters in a .pth checkpoint")
    parser.add_argument(
        "checkpoint",
        type=str,
        default="/mnt/adas7tb/jgu/Orion/adzoo/orion/work_dirs/orion_student_exp1_eva_init/iter_5502.pth",
        nargs="?",
        help="Path to checkpoint .pth file",
    )
    args = parser.parse_args()

    try:
        result = count_params(args.checkpoint)
    except FileNotFoundError:
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tensors (state_dict keys): {result['tensor_count']}")
    print(f"Total parameters: {result['total_params']:,}")
    print(f"Total parameters (M): {result['total_params_M']:.4f} M")


if __name__ == "__main__":
    main()
