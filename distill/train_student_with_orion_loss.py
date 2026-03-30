import os
import glob
import json
import logging
import argparse
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Ensure the distill/ directory is on the path so local modules are importable
# regardless of where the script is invoked from.
_DISTILL_DIR = os.path.dirname(os.path.abspath(__file__))
if _DISTILL_DIR not in sys.path:
    sys.path.insert(0, _DISTILL_DIR)


class NpzDistillDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        if not self.files:
            raise RuntimeError(f'No npz files found in {data_dir}')
        print(f'Loaded {len(self.files)} distill samples from {data_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path) as d:
            vision_embeded = d['visual_queries']
            planning_token = d['planning_token']
            ego_fut_trajs = d['ego_fut_trajs']
            ego_fut_masks = d['ego_fut_masks']
            ego_fut_cmd   = d['ego_fut_cmd']
            lane_scores   = d['lane_scores']
            lane_preds    = d['lane_preds']
            
            agent_preds_np = d['agent_preds'] if 'agent_preds' in d.files else None
            agent_fut_preds_np = d['agent_fut_preds'] if 'agent_fut_preds' in d.files else None
            agent_score_preds_np = d['agent_score_preds'] if 'agent_score_preds' in d.files else None
            agent_fut_cls_preds_np = d['agent_fut_cls_preds'] if 'agent_fut_cls_preds' in d.files else None

        vision_embeded = torch.from_numpy(vision_embeded).float()
        planning_token = torch.from_numpy(planning_token).float()
        ego_fut_trajs = torch.from_numpy(ego_fut_trajs).float()
        ego_fut_masks = torch.from_numpy(ego_fut_masks).float()
        ego_fut_cmd = torch.from_numpy(ego_fut_cmd).float()
        lane_scores = torch.from_numpy(lane_scores).float()
        lane_preds = torch.from_numpy(lane_preds).float()
        
        agent_preds = torch.from_numpy(agent_preds_np).float() if agent_preds_np is not None else None
        agent_fut_preds = torch.from_numpy(agent_fut_preds_np).float() if agent_fut_preds_np is not None else None
        agent_score_preds = torch.from_numpy(agent_score_preds_np).float() if agent_score_preds_np is not None else None
        agent_fut_cls_preds = torch.from_numpy(agent_fut_cls_preds_np).float() if agent_fut_cls_preds_np is not None else None

        vision_embeded = vision_embeded.squeeze(0)
        planning_token = planning_token.squeeze(0)
        ego_fut_trajs = ego_fut_trajs.squeeze(0)

        sample = {
            'vision_embeded': vision_embeded,
            'target_ego_feature': planning_token[0],
            'ego_fut_trajs': ego_fut_trajs,
            'ego_fut_masks': ego_fut_masks,
            'ego_fut_cmd': ego_fut_cmd,
            'lane_scores': lane_scores,
            'lane_preds': lane_preds,
        }
        if agent_preds is not None:
            if agent_preds.dim() > 0 and agent_preds.shape[0] == 1:
                agent_preds = agent_preds.squeeze(0)
            sample['agent_preds'] = agent_preds
        if agent_fut_preds is not None:
            if agent_fut_preds.dim() > 0 and agent_fut_preds.shape[0] == 1:
                agent_fut_preds = agent_fut_preds.squeeze(0)
            sample['agent_fut_preds'] = agent_fut_preds
        if agent_score_preds is not None:
            if agent_score_preds.dim() > 0 and agent_score_preds.shape[0] == 1:
                agent_score_preds = agent_score_preds.squeeze(0)
            sample['agent_score_preds'] = agent_score_preds
        if agent_fut_cls_preds is not None:
            if agent_fut_cls_preds.dim() > 0 and agent_fut_cls_preds.shape[0] == 1:
                agent_fut_cls_preds = agent_fut_cls_preds.squeeze(0)
            sample['agent_fut_cls_preds'] = agent_fut_cls_preds
        return sample

def collate_fn(batch):
    return torch.utils.data._utils.collate.default_collate(batch)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--train-dir', type=str, default='distill_data/train')
    parser.add_argument('--val-dir', type=str, default='distill_data/val')
    
    parser.add_argument('--hidden-dim', type=int, default=1024)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--input-dim', type=int, default=4096)
    parser.add_argument('--output-dim', type=int, default=4096)
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--val-batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    
    parser.add_argument('--use-feature-mimic-loss', action='store_true')
    parser.add_argument('--with-bound-loss', action='store_true', default=True)
    parser.add_argument('--use-col-loss', action='store_true', default=True)
    # Reduce open-loop overfitting / improve closed-loop (see METHODS_CLOSED_LOOP.md)
    parser.add_argument('--vision-noise-std', type=float, default=0.0,
                        help='Add Gaussian noise to vision_embeded during train (e.g. 0.01–0.05)')
    parser.add_argument('--mimic-weight-decay-epochs', type=int, default=0,
                        help='Decay mimic loss weight to 0 over this many epochs (0 = no decay)')
    parser.add_argument('--mimic-loss', type=str, default='l1',
                        choices=['l1', 'l2', 'kl', 'huber'],
                        help='Feature mimic loss type')
    parser.add_argument('--ablate-mimic-loss', action='store_true',
                        help='Run ablation for feature mimic loss (l2, kl, huber)')
    
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--dist-backend', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--ckpt-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save-every-epochs', type=int, default=0)
    
    # Add new argument for Orion checkpoint
    parser.add_argument('--orion-ckpt', type=str, default=None,
                        help='Path to original Orion checkpoint to load VAE weights')
    
    return parser.parse_args()

def set_seed(seed: int):
    if seed <= 0: return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def is_distributed_run():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def init_distributed(args):
    if not is_distributed_run():
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        return

    args.distributed = True
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = getattr(args, "dist_backend", None) or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend, init_method="env://")

def setup_device(args):
    if args.distributed and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        return torch.device("cuda", args.local_rank)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(args.device if torch.cuda.is_available() else "cpu")

def setup_logger(log_path, rank=0):
    logger = logging.getLogger(f"distill_train.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s" if rank == 0 else f"%(asctime)s - [rank {rank}] %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def save_checkpoint(path, model, optimizer, epoch, global_step, best_val_loss, args, val_loss=None):
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "val_loss": val_loss,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, path)

def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt

def reduce_losses(losses, world_size, device):
    if world_size <= 1:
        return losses
    reduced = {}
    for k, v in losses.items():
        t = torch.tensor([v], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        reduced[k] = (t / world_size).item()
    return reduced

def train_one(args, train_loader, val_loader, train_sampler, device, mimic_loss_type, run_dir):
    log_dir, ckpt_dir = os.path.join(run_dir, "logs"), os.path.join(run_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    run_args = dict(vars(args))
    run_args["mimic_loss"] = mimic_loss_type
    if args.rank == 0:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(run_args, f, indent=2, sort_keys=True)
    
    logger = setup_logger(os.path.join(log_dir, "train.log"), rank=args.rank)
    if args.rank == 0:
        logger.info(f"Config: {run_args}")
        logger.info(f"=== Mimic Loss: {mimic_loss_type} ===")

    from student_model_with_orion_loss import OrionMimicModel
    model = OrionMimicModel(
        input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim,
        num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
        with_bound_loss=args.with_bound_loss, use_col_loss=args.use_col_loss,
        mimic_loss_type=mimic_loss_type,
    ).to(device)
    
    # Load VAE weights if path is provided
    if args.orion_ckpt:
        if args.rank == 0:
            logger.info(f"Loading Orion VAE weights from {args.orion_ckpt}")
        model.load_orion_vae_weights(args.orion_ckpt, device=device)

    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None, find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 0

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        logger.info(f"Resumed from {args.resume}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if train_sampler: train_sampler.set_epoch(epoch)
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=args.rank!=0)
        epoch_losses = {}
        
        for step, batch in enumerate(train_bar):
            # Move to device
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            agent_outs = {k: batch_gpu[k] for k in ('agent_preds', 'agent_fut_preds', 'agent_score_preds', 'agent_fut_cls_preds') if k in batch_gpu} if 'agent_preds' in batch else None
            
            # Squeeze if needed
            lane_preds = batch_gpu['lane_preds']
            lane_scores = batch_gpu['lane_scores']
            if lane_preds.dim() == 5 and lane_preds.shape[1] == 1: lane_preds = lane_preds.squeeze(1)
            if lane_scores.dim() == 4 and lane_scores.shape[1] == 1: lane_scores = lane_scores.squeeze(1)

            vision_in = batch_gpu['vision_embeded']
            if getattr(args, 'vision_noise_std', 0) > 0:
                vision_in = vision_in + torch.randn_like(vision_in, device=vision_in.device) * args.vision_noise_std

            outputs = model(
                vision_embeded=vision_in,
                ego_fut_trajs=batch_gpu['ego_fut_trajs'],
                ego_fut_masks=batch_gpu['ego_fut_masks'],
                ego_fut_cmd=batch_gpu['ego_fut_cmd'],
                lane_preds=lane_preds,
                lane_scores=lane_scores,
                agent_outs=agent_outs,
                target_ego_feature=batch_gpu['target_ego_feature'],
                return_loss=True,
            )
            losses = outputs['losses']
            mimic_weight = 1.0
            if getattr(args, 'mimic_weight_decay_epochs', 0) > 0 and 'loss_feature_mimic' in losses:
                mimic_weight = max(0.0, 1.0 - (epoch + 1) / args.mimic_weight_decay_epochs)
            total_loss = 0.0
            for k, v in losses.items():
                if k == 'loss_feature_mimic':
                    total_loss = total_loss + (mimic_weight * v if args.use_feature_mimic_loss else 0.0)
                else:
                    total_loss = total_loss + v

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            global_step += 1

            # Logging
            loss_scalars = {k: v.item() for k, v in losses.items()}
            loss_scalars['total'] = total_loss.item()
            for k, v in loss_scalars.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            
            if args.rank == 0:
                postfix = {"loss": f"{total_loss.item():.4f}"}
                for k in ['loss_feature_mimic', 'loss_plan_reg', 'loss_plan_bound', 'loss_plan_col', 'loss_vae_gen']:
                    if k in losses:
                        short_k = k.replace('loss_', '').replace('feature_', '').replace('plan_', '')
                        postfix[short_k] = f"{losses[k].item():.4f}"
                train_bar.set_postfix(postfix)

        # Epoch Summary
        avg_train_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        if args.distributed:
            avg_train_losses = reduce_losses(avg_train_losses, args.world_size, device)
        
        if args.rank == 0:
            logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_losses['total']:.4f}")
            logger.info(f"Train Details: {avg_train_losses}")

        # Validation
        if args.rank == 0 and val_loader:
            model.eval()
            val_losses = {}
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    agent_outs = {k: batch_gpu[k] for k in ('agent_preds', 'agent_fut_preds', 'agent_score_preds', 'agent_fut_cls_preds') if k in batch_gpu} if 'agent_preds' in batch else None
                    lane_preds, lane_scores = batch_gpu['lane_preds'], batch_gpu['lane_scores']
                    if lane_preds.dim() == 5 and lane_preds.shape[1] == 1: lane_preds = lane_preds.squeeze(1)
                    if lane_scores.dim() == 4 and lane_scores.shape[1] == 1: lane_scores = lane_scores.squeeze(1)

                    outputs = model(
                        vision_embeded=batch_gpu['vision_embeded'],
                        ego_fut_trajs=batch_gpu['ego_fut_trajs'],
                        ego_fut_masks=batch_gpu['ego_fut_masks'],
                        ego_fut_cmd=batch_gpu['ego_fut_cmd'],
                        lane_preds=lane_preds,
                        lane_scores=lane_scores,
                        agent_outs=agent_outs,
                        target_ego_feature=batch_gpu['target_ego_feature'],
                        return_loss=True,
                    )
                    losses = outputs['losses']
                    total_loss = sum(losses.values()) if args.use_feature_mimic_loss else sum(v for k, v in losses.items() if k != 'loss_feature_mimic')
                    
                    loss_scalars = {k: v.item() for k, v in losses.items()}
                    loss_scalars['total'] = total_loss.item()
                    for k, v in loss_scalars.items():
                        val_losses[k] = val_losses.get(k, 0.0) + v
            
            avg_val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}
            logger.info(f"Epoch {epoch+1} Val Loss: {avg_val_losses['total']:.4f}")
            logger.info(f"Val Details: {avg_val_losses}")

            if avg_val_losses['total'] < best_val_loss:
                best_val_loss = avg_val_losses['total']
                save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, optimizer, epoch+1, global_step, best_val_loss, args)
                logger.info("Saved best checkpoint")
            save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optimizer, epoch+1, global_step, best_val_loss, args)
        
        if args.distributed:
            dist.barrier()

    if args.rank == 0: logger.info("Training completed!")
    if args.distributed: dist.barrier()

def main():
    args = parse_args()
    init_distributed(args)
    set_seed(args.seed + (args.rank if args.seed > 0 else 0))
    device = setup_device(args)

    train_ds = NpzDistillDataset(args.train_dir)
    val_ds = NpzDistillDataset(args.val_dir)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if args.distributed else None
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    ) if not args.distributed or args.rank == 0 else None

    script_dir = os.path.dirname(__file__)
    out_dir = args.out_dir or os.path.join(script_dir, "results")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_dir = os.path.join(out_dir, run_name)
    os.makedirs(base_run_dir, exist_ok=True)

    if args.ablate_mimic_loss:
        for mimic_loss_type in ["l2", "kl", "huber"]:
            run_dir = os.path.join(base_run_dir, mimic_loss_type)
            train_one(args, train_loader, val_loader, train_sampler, device, mimic_loss_type, run_dir)
    else:
        train_one(args, train_loader, val_loader, train_sampler, device, args.mimic_loss, base_run_dir)

    if args.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
