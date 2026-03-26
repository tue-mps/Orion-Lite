import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import argparse
import numpy as np

# Import your modules
from dataloader import get_dataloaders
from student_model import OrionStudent

# [DEBUG] Enable Anomaly Detection to find where NaNs originate in the backward pass
torch.autograd.set_detect_anomaly(True)

def check_for_nans(tensor, name, filenames):
    """Checks a tensor for NaNs or Infs and prints the corresponding filename if found."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"\n[!] ERROR: NaN or Inf detected in {name}!")
        # Check each item in the batch
        for i in range(tensor.shape[0]):
            if torch.isnan(tensor[i]).any() or torch.isinf(tensor[i]).any():
                print(f"    -> Corrupt File: {filenames[i]}")
        return True
    return False

def build_loss(loss_name):
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=1.0)
    if loss_name == "kl":
        return nn.KLDivLoss(reduction="batchmean")
    raise ValueError(f"Unsupported loss: {loss_name}")

def compute_loss(loss_name, criterion, student_preds, teacher_targets):
    if loss_name == "kl":
        log_probs = F.log_softmax(student_preds, dim=-1)
        target_probs = F.softmax(teacher_targets, dim=-1)
        return criterion(log_probs, target_probs)
    return criterion(student_preds, teacher_targets)

def train_one(args, loss_name, checkpoint_dir):
    # -----------------------------------------------------------------------------
    # 1. Setup Device & Directories
    # -----------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # -----------------------------------------------------------------------------
    # 2. Prepare Data
    # -----------------------------------------------------------------------------
    print("Initializing Dataloaders...")
    train_loader, val_loader = get_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # -----------------------------------------------------------------------------
    # 3. Initialize Model
    # -----------------------------------------------------------------------------
    print("Initializing Student Model...")
    model = OrionStudent(
        input_dim=4096, 
        hidden_dim=args.hidden_dim,
        output_dim=4096,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    # -----------------------------------------------------------------------------
    # 4. Setup Optimizer & Loss
    # -----------------------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    criterion = build_loss(loss_name)

    # -----------------------------------------------------------------------------
    # 5. Training Loop
    # -----------------------------------------------------------------------------
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss_accum = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch in progress_bar:
            # Move data to device
            visual_queries = batch['vision_embeded'].to(device)
            teacher_targets = batch['current_states'].to(device)
            filenames = batch['filename'] # Get filenames for debugging

            # [DEBUG] Check Inputs for NaNs BEFORE Forward Pass
            if check_for_nans(visual_queries, "Input (visual_queries)", filenames):
                print("Aborting training due to Bad Input Data.")
                return
            if check_for_nans(teacher_targets, "Target (current_states)", filenames):
                print("Aborting training due to Bad Target Data.")
                return

            # Forward Pass
            student_preds = model(visual_queries)
            
            # [DEBUG] Check Output for NaNs (Model Stability Issue)
            if torch.isnan(student_preds).any():
                print(f"\n[!] ERROR: Model produced NaNs! (Exploding Gradients?)")
                return

            # Compute Loss
            loss = compute_loss(loss_name, criterion, student_preds, teacher_targets)
            
            # [DEBUG] Check Loss
            if torch.isnan(loss):
                print(f"\n[!] ERROR: Loss is NaN!")
                return

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Logging
            train_loss_accum += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # -----------------------------------------------------------------------------
        # 6. Validation Loop
        # -----------------------------------------------------------------------------
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                visual_queries = batch['vision_embeded'].to(device)
                teacher_targets = batch['current_states'].to(device)
                
                student_preds = model(visual_queries)
                val_loss = compute_loss(loss_name, criterion, student_preds, teacher_targets)
                val_loss_accum += val_loss.item()
        
        avg_val_loss = val_loss_accum / len(val_loader)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  LR:         {current_lr:.2e}")

        # -----------------------------------------------------------------------------
        # 7. Checkpointing
        # -----------------------------------------------------------------------------
        if avg_val_loss < best_val_loss:
            print(f"  [!] Validation Loss improved ({best_val_loss:.6f} -> {avg_val_loss:.6f}). Saving model...")
            best_val_loss = avg_val_loss
            save_path = os.path.join(checkpoint_dir, "best_student_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': vars(args)
            }, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Orion Student Model via Distillation")
    
    # Data Paths
    parser.add_argument('--train_dir', type=str, default='distill_data/train', help='Path to training .npz files')
    parser.add_argument('--val_dir', type=str, default='distill_data/val', help='Path to validation .npz files')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_student', help='Directory to save checkpoints')
    
    # Model Hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension of student transformer')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer decoder layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of dataloader workers')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'kl', 'l1', 'huber'], help='Loss function to use')
    parser.add_argument('--ablate', action='store_true', help='Run ablation across mse, kl, l1, huber')

    args = parser.parse_args()
    
    if args.ablate:
        # for loss_name in ["mse", "kl", "l1", "huber"]:
        for loss_name in ["kl", "l1", "huber"]:
            print(f"\n=== Loss: {loss_name} ===")
            loss_checkpoint_dir = os.path.join(args.checkpoint_dir, loss_name)
            train_one(args, loss_name, loss_checkpoint_dir)
    else:
        train_one(args, args.loss, args.checkpoint_dir)
