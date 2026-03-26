import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DistillationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing .npz files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.transform = transform
        
        if len(self.file_paths) == 0:
            print(f"Warning: No .npz files found in {data_dir}")
        else:
            print(f"Found {len(self.file_paths)} files in {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.file_paths[idx]
        
        try:
            # Load npz file
            # Using mmap_mode='r' can be faster for large files if we only read parts, 
            # but np.load for npz doesn't support mmap directly on the zip container easily 
            # without extraction. Standard load is fine for these file sizes (~3.8MB).
            with np.load(file_path) as data:
                # Inputs: visual_queries (B, N, 4096) -> We expect B=1 per file usually
                # Shapes in file: (1, 513, 4096)
                vision_embeded = data['visual_queries']
                
                # Targets: planning_token (B, 1, 4096)
                current_states = data['planning_token']

            # Squeeze batch dim if it exists (files saved as [1, ...])
            # DataLoader will add a new batch dimension.
            if vision_embeded.ndim == 3 and vision_embeded.shape[0] == 1:
                vision_embeded = vision_embeded[0]
            if current_states.ndim == 3 and current_states.shape[0] == 1:
                current_states = current_states[0]

            # Convert to Float Tensor (Student model usually trains in FP32 or BF16)
            # Data is saved as float16
            vision_embeded = torch.from_numpy(vision_embeded).float()
            current_states = torch.from_numpy(current_states).float()

            sample = {
                'vision_embeded': vision_embeded, 
                'current_states': current_states,
                'filename': os.path.basename(file_path)
            }

            if self.transform:
                sample = self.transform(sample)

            return sample

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a dummy zero tensor or handle gracefully
            # Here we raise to stop and fix data issues
            raise e

def get_dataloaders(train_dir, val_dir, batch_size=32, num_workers=4):
    """
    Creates training and validation DataLoaders.
    """
    train_dataset = DistillationDataset(train_dir)
    val_dataset = DistillationDataset(val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataloader
    train_path = "/mnt/adas7tb/jgu/Orion/data/distill_data/train"
    # Assuming you have a val directory, otherwise use train for test
    val_path = "/mnt/adas7tb/jgu/Orion/data/distill_data/val" 
    
    # Create dummy val dir if it doesn't exist for testing
    if not os.path.exists(val_path):
        val_path = train_path

    train_loader, val_loader = get_dataloaders(train_path, val_path, batch_size=4)

    print("Checking Train Loader:")
    for i, batch in enumerate(train_loader):
        inputs = batch['vision_embeded']
        targets = batch['current_states']
        print(f"Batch {i}: Input {inputs.shape}, Target {targets.shape}")
        break