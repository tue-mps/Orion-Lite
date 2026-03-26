import os
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_student_with_orion_loss import (
    NpzDistillDataset,
    collate_fn,
    setup_logger,
    save_checkpoint,
    load_checkpoint,
)
from student_model_with_orion_loss import OrionMimicModel


def ensure_conda_include_paths():
    """Ensure conda include dir (for crypt.h, etc.) is on compiler search paths."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    include_dir = os.path.join(conda_prefix, "include")
    for var in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        existing = os.environ.get(var, "")
        paths = existing.split(os.pathsep) if existing else []
        if include_dir not in paths:
            paths = [include_dir] + paths if existing else [include_dir]
            os.environ[var] = os.pathsep.join(paths)


def parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(v) for v in str(value).split(",") if v != ""]


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--train-dir", type=str, default="distill_data/train")
    parser.add_argument("--val-dir", type=str, default="distill_data/val")

    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--output-dim", type=int, default=4096)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--use-feature-mimic-loss", action="store_true")
    parser.add_argument("--with-bound-loss", action="store_true", default=True)
    parser.add_argument("--use-col-loss", action="store_true", default=True)

    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--orion-ckpt", type=str, default=None,
                        help="Path to original Orion checkpoint to load VAE weights (optional)")
    parser.add_argument("--save-every-epochs", type=int, default=0)
    parser.add_argument("--compile", action="store_true",
                        help="Wrap models with torch.compile for potential speedups.")

    parser.add_argument("--layers", type=str, default="2,4,8,16",
                        help="Comma-separated decoder layer counts.")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="Comma-separated GPU ids (same length as layers).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint. Provide a file path, a run dir (will use checkpoints/last.pt), or 'auto' to use each run_dir/checkpoints/last.pt.")

    return parser.parse_args()


def set_seed(seed: int):
    if seed <= 0:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def compute_losses(model, batch_gpu, args):
    agent_outs = None
    if "agent_preds" in batch_gpu:
        agent_outs = {k: batch_gpu[k] for k in
                      ("agent_preds", "agent_fut_preds", "agent_score_preds", "agent_fut_cls_preds")
                      if k in batch_gpu}

    lane_preds = batch_gpu["lane_preds"]
    lane_scores = batch_gpu["lane_scores"]
    if lane_preds.dim() == 5 and lane_preds.shape[1] == 1:
        lane_preds = lane_preds.squeeze(1)
    if lane_scores.dim() == 4 and lane_scores.shape[1] == 1:
        lane_scores = lane_scores.squeeze(1)

    outputs = model(
        vision_embeded=batch_gpu["vision_embeded"],
        ego_fut_trajs=batch_gpu["ego_fut_trajs"],
        ego_fut_masks=batch_gpu["ego_fut_masks"],
        ego_fut_cmd=batch_gpu["ego_fut_cmd"],
        lane_preds=lane_preds,
        lane_scores=lane_scores,
        agent_outs=agent_outs,
        target_ego_feature=batch_gpu["target_ego_feature"],
        return_loss=True,
    )
    losses = outputs["losses"]
    if args.use_feature_mimic_loss:
        total_loss = sum(losses.values())
    else:
        total_loss = sum(v for k, v in losses.items() if k != "loss_feature_mimic")
    return total_loss, losses


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_conda_include_paths()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    layers = parse_int_list(args.layers)
    gpus = parse_int_list(args.gpus)
    if not layers or not gpus or len(layers) != len(gpus):
        raise ValueError("`layers` and `gpus` must be non-empty and the same length.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-GPU ablation training.")

    train_ds = NpzDistillDataset(args.train_dir)
    val_ds = NpzDistillDataset(args.val_dir)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=args.num_workers > 0, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=args.num_workers > 0, prefetch_factor=4,
    )

    script_dir = os.path.dirname(__file__)
    out_dir = args.out_dir or os.path.join(script_dir, "runs_ablation_layers")
    os.makedirs(out_dir, exist_ok=True)
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    models = []
    optimizers = []
    devices = []
    loggers = []
    run_dirs = []
    best_val_losses = []
    global_steps = []
    start_epochs = []

    for idx, (layer, gpu) in enumerate(zip(layers, gpus)):
        device = torch.device(f"cuda:{gpu}")
        devices.append(device)
        model = OrionMimicModel(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=layer,
            num_heads=args.num_heads,
            dropout=args.dropout,
            with_bound_loss=args.with_bound_loss,
            use_col_loss=args.use_col_loss,
        ).to(device)

        if args.compile:
            try:
                model = torch.compile(model)
            except Exception as e:
                print(f"torch.compile failed for layer {layer} on gpu {gpu}: {e}. Falling back to eager.", flush=True)

        if args.orion_ckpt:
            model.load_orion_vae_weights(args.orion_ckpt, device=device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        models.append(model)
        optimizers.append(optimizer)
        best_val_losses.append(float("inf"))
        global_steps.append(0)

        run_name = f"{run_id}_decoder{layer}l_gpu{gpu}"
        run_dir = os.path.join(out_dir, run_name)
        run_dirs.append(run_dir)
        log_dir = os.path.join(run_dir, "logs")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        cfg = vars(args).copy()
        cfg["num_layers"] = layer
        cfg["gpu"] = gpu
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)

        logger = setup_logger(
            os.path.join(log_dir, "train.log"),
            rank=f"layer{layer}_gpu{gpu}",
        )
        logger.info(f"Config: {cfg}")
        loggers.append(logger)

        resume_path = None
        if args.resume:
            if args.resume.lower() == "auto":
                candidate = os.path.join(ckpt_dir, "last.pt")
                if os.path.exists(candidate):
                    resume_path = candidate
            elif os.path.isdir(args.resume):
                candidate = os.path.join(args.resume, "checkpoints", "last.pt")
                if os.path.exists(candidate):
                    resume_path = candidate
            elif os.path.isfile(args.resume):
                resume_path = args.resume

        start_epoch = 0
        if resume_path:
            try:
                ckpt = load_checkpoint(resume_path, model, optimizer, device)
                start_epoch = ckpt.get("epoch", 0)
                global_steps[idx] = ckpt.get("global_step", 0)
                best_val_losses[idx] = ckpt.get("best_val_loss", float("inf"))
                logger.info(f"Resumed from {resume_path} at epoch {start_epoch}, step {global_steps[idx]}")
            except Exception as e:
                logger.error(f"Failed to resume from {resume_path}: {e}. Starting from scratch.")

        start_epochs.append(start_epoch)

    # Launch compute for each model on its own CUDA stream so steps can overlap across GPUs.
    streams = [torch.cuda.Stream(device=device) for device in devices]

    min_start_epoch = min(start_epochs) if start_epochs else 0

    for epoch in range(min_start_epoch, args.epochs):
        for model in models:
            model.train()

        epoch_losses = [dict() for _ in models]
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in train_bar:
            batch_results = []
            for i, (model, optimizer, device, stream) in enumerate(zip(models, optimizers, devices, streams)):
                if epoch < start_epochs[i]:
                    continue
                with torch.cuda.stream(stream):
                    torch.cuda.set_device(device)
                    batch_gpu = move_batch_to_device(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    total_loss, losses = compute_losses(model, batch_gpu, args)

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    global_steps[i] += 1

                    detached_losses = {k: v.detach() for k, v in losses.items()}
                    batch_results.append((i, detached_losses, total_loss.detach()))

            for stream in streams:
                stream.synchronize()

            for i, losses, total_loss in batch_results:
                loss_scalars = {k: v.item() for k, v in losses.items()}
                loss_scalars["total"] = total_loss.item()
                for k, v in loss_scalars.items():
                    epoch_losses[i][k] = epoch_losses[i].get(k, 0.0) + v

        for i, logger in enumerate(loggers):
            if epoch < start_epochs[i]:
                logger.info(f"Epoch {epoch + 1} skipped for this model (resume starts at epoch {start_epochs[i]}).")
                continue
            avg_train = {k: v / len(train_loader) for k, v in epoch_losses[i].items()}
            logger.info(f"Epoch {epoch + 1} Train Loss: {avg_train.get('total', 0.0):.4f}")
            logger.info(f"Train Details: {avg_train}")

        for model in models:
            model.eval()

        val_losses = [dict() for _ in models]
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch_results = []
                for i, (model, device, stream) in enumerate(zip(models, devices, streams)):
                    if epoch < start_epochs[i]:
                        continue
                    with torch.cuda.stream(stream):
                        torch.cuda.set_device(device)
                        batch_gpu = move_batch_to_device(batch, device)
                        total_loss, losses = compute_losses(model, batch_gpu, args)
                        batch_results.append((i, {k: v.detach() for k, v in losses.items()}, total_loss.detach()))

                for stream in streams:
                    stream.synchronize()

                for i, losses, total_loss in batch_results:
                    loss_scalars = {k: v.item() for k, v in losses.items()}
                    loss_scalars["total"] = total_loss.item()
                    for k, v in loss_scalars.items():
                        val_losses[i][k] = val_losses[i].get(k, 0.0) + v

        for i, (logger, run_dir) in enumerate(zip(loggers, run_dirs)):
            if epoch < start_epochs[i]:
                continue
            avg_val = {k: v / len(val_loader) for k, v in val_losses[i].items()}
            logger.info(f"Epoch {epoch + 1} Val Loss: {avg_val.get('total', 0.0):.4f}")
            logger.info(f"Val Details: {avg_val}")

            ckpt_dir = os.path.join(run_dir, "checkpoints")
            if avg_val.get("total", float("inf")) < best_val_losses[i]:
                best_val_losses[i] = avg_val["total"]
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pt"),
                    models[i],
                    optimizers[i],
                    epoch + 1,
                    global_steps[i],
                    best_val_losses[i],
                    args,
                )
                logger.info("Saved best checkpoint")
            save_checkpoint(
                os.path.join(ckpt_dir, "last.pt"),
                models[i],
                optimizers[i],
                epoch + 1,
                global_steps[i],
                best_val_losses[i],
                args,
            )
            if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
                save_checkpoint(
                    os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt"),
                    models[i],
                    optimizers[i],
                    epoch + 1,
                    global_steps[i],
                    best_val_losses[i],
                    args,
                )

    for logger in loggers:
        logger.info("Training completed!")


if __name__ == "__main__":
    main()
