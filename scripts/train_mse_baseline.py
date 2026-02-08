"""
TabTransformer MSE Baseline for PGC Survival Prediction.

Training on all data from training folder, evaluating on inference folder.
No validation set, fixed number of epochs.
Uses MSE loss for survival time regression.
"""

import os
import argparse
import glob
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.continuous_features import CONTINUOUS_FEATURES
from src.data.dataset_v2 import (
    PGCDatasetV2,
    collate_fn as collate_fn_v2,
    COMPUTED_FEATURES,
)
from src.models import TransformerBackbone
from src.models.heads import get_head
from src.training.losses import get_loss_fn


# Constants
TARGET_SCALE = 2000.0
NUM_PHASES = 10
SAMPLES_PER_PHASE = 5


def get_all_files_from_folder(folder_path: str, toy_ratio: float = 1.0) -> List[str]:
    """Get all CSV file names from a folder."""
    import random
    
    csv_paths = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    csv_files = [os.path.basename(p) for p in csv_paths]
    
    print(f"Found {len(csv_files)} CSV files in {folder_path}")
    
    if toy_ratio < 1.0:
        random.seed(42)
        n_files = max(1, int(len(csv_files) * toy_ratio))
        csv_files = random.sample(csv_files, n_files)
        print(f"Toy mode: using {n_files} files ({toy_ratio*100:.0f}%)")
    
    return csv_files


def create_dataloader(
    folder_path: str,
    file_list: List[str],
    scaler_params: Dict = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> Tuple[DataLoader, Dict]:
    """
    Create DataLoader for training or inference.
    
    Returns:
        Tuple of (DataLoader, scaler_params)
    """
    dataset = PGCDatasetV2(
        folder_path=folder_path,
        file_list=file_list,
        continuous_features=CONTINUOUS_FEATURES,
        scaler_params=scaler_params,
    )
    
    # Compute scaler if not provided
    if scaler_params is None:
        scaler_params = dataset.compute_scaler_params()
        dataset.scaler_params = scaler_params
        dataset._apply_scaling()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_v2,
    )
    
    return dataloader, scaler_params


class SimpleTrainer:
    """
    Simple trainer for MSE baseline (no validation, fixed epochs).
    """
    
    def __init__(
        self,
        backbone: TransformerBackbone,
        head: nn.Module,
        train_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = device
        self.backbone = backbone.to(device)
        self.head = head.to(device)
        self.train_loader = train_loader
        
        # Loss function (MSE)
        self.criterion = get_loss_fn("mse")
        
        # Optimizer
        params = list(self.backbone.parameters()) + list(self.head.parameters())
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        
        # Training history
        self.history = {"train_loss": []}
    
    def _forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass."""
        x = batch["x_continuous"].to(self.device)
        positions = batch["positions"].to(self.device)
        bluezone = batch["bluezone_info"].to(self.device)
        whitezone = batch["whitezone_info"].to(self.device)
        map_idx = batch["map_idx"].to(self.device)
        
        embeddings = self.backbone(x, positions, bluezone, whitezone, map_idx)
        predictions = self.head(embeddings)
        
        return predictions
    
    def train_epoch(self, epoch: int, num_epochs: int) -> float:
        """Train for one epoch."""
        self.backbone.train()
        self.head.train()
        
        total_loss = 0.0
        num_batches = 0
        
        loader = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False,
        )
        
        for batch in loader:
            y = batch["y"].to(self.device)
            alive_mask = batch["squad_alive_mask"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            
            self.optimizer.zero_grad()
            pred = self._forward(batch)
            
            # Compute masked loss
            valid_mask = alive_mask & padding_mask
            loss = self.criterion(pred, y, valid_mask)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            loader.set_postfix(loss=f"{total_loss/num_batches:.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs: int) -> Dict:
        """Train for fixed number of epochs."""
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch, num_epochs)
            self.history["train_loss"].append(train_loss)
            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(self, save_path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            "backbone_state_dict": self.backbone.state_dict(),
            "head_state_dict": self.head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to: {save_path}")


def evaluate_on_inference(
    backbone: TransformerBackbone,
    head: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict:
    """
    Evaluate model on inference data.
    
    Returns:
        Dictionary with phase-wise metrics.
    """
    from scipy.special import softmax
    
    backbone = backbone.to(device)
    head = head.to(device)
    backbone.eval()
    head.eval()
    
    # Collect predictions
    all_preds = []
    all_targets = []
    all_time_points = []
    all_match_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            x = batch["x_continuous"].to(device)
            positions = batch["positions"].to(device)
            bluezone = batch["bluezone_info"].to(device)
            whitezone = batch["whitezone_info"].to(device)
            map_idx = batch["map_idx"].to(device)
            y = batch["y"]
            time_points = batch["time_point"]
            padding_mask = batch["padding_mask"]
            
            embeddings = backbone(x, positions, bluezone, whitezone, map_idx)
            predictions = head(embeddings).cpu()
            
            batch_size = x.size(0)
            for i in range(batch_size):
                valid_mask = padding_mask[i].bool()
                pred = predictions[i][valid_mask].numpy()
                target = y[i][valid_mask].numpy()
                tp = time_points[i].item()
                
                all_preds.append(pred)
                all_targets.append(target)
                all_time_points.append(tp)
    
    # Group by time_point and compute metrics
    # Sort time_points and assign phases
    unique_time_points = sorted(set(all_time_points))
    tp_to_phase = {
        tp: min(idx // SAMPLES_PER_PHASE + 1, NUM_PHASES)
        for idx, tp in enumerate(unique_time_points)
    }
    
    phase_data = defaultdict(list)
    
    for pred, target, tp in zip(all_preds, all_targets, all_time_points):
        phase = tp_to_phase.get(tp, NUM_PHASES)
        phase_data[phase].append({
            'preds': pred,
            'targets': target,
        })
    
    # Compute metrics per phase
    results = {
        'accuracy': {},
        'log_loss': {},
        'ece': {},
    }
    
    all_correct = 0
    all_total = 0
    all_log_losses = []
    all_confidences = []
    all_accuracies = []
    
    for phase in range(1, NUM_PHASES + 1):
        if phase not in phase_data:
            for metric in results:
                results[metric][f'Phase_{phase}'] = 0.0
            continue
        
        phase_correct = 0
        phase_total = 0
        phase_log_losses = []
        phase_confidences = []
        phase_accuracies = []
        
        for sample in phase_data[phase]:
            preds = sample['preds']
            targets = sample['targets']
            
            if len(preds) < 2:
                continue
            
            pred_winner = np.argmax(preds)
            target_winner = np.argmax(targets)
            
            is_correct = pred_winner == target_winner
            if is_correct:
                phase_correct += 1
            phase_total += 1
            
            probs = softmax(preds)
            winner_prob = probs[target_winner]
            
            winner_prob_clipped = np.clip(winner_prob, 1e-7, 1 - 1e-7)
            phase_log_losses.append(-np.log(winner_prob_clipped))
            
            confidence = np.max(probs)
            phase_confidences.append(confidence)
            phase_accuracies.append(float(is_correct))
        
        if phase_total > 0:
            results['accuracy'][f'Phase_{phase}'] = phase_correct / phase_total
        else:
            results['accuracy'][f'Phase_{phase}'] = 0.0
        
        if phase_log_losses:
            results['log_loss'][f'Phase_{phase}'] = np.mean(phase_log_losses)
        else:
            results['log_loss'][f'Phase_{phase}'] = 0.0
        
        if phase_confidences:
            results['ece'][f'Phase_{phase}'] = compute_ece(
                np.array(phase_confidences),
                np.array(phase_accuracies),
            )
        else:
            results['ece'][f'Phase_{phase}'] = 0.0
        
        all_correct += phase_correct
        all_total += phase_total
        all_log_losses.extend(phase_log_losses)
        all_confidences.extend(phase_confidences)
        all_accuracies.extend(phase_accuracies)
    
    # Average metrics
    if all_total > 0:
        results['accuracy']['Average'] = all_correct / all_total
    else:
        results['accuracy']['Average'] = 0.0
    
    if all_log_losses:
        results['log_loss']['Average'] = np.mean(all_log_losses)
    else:
        results['log_loss']['Average'] = 0.0
    
    if all_confidences:
        results['ece']['Average'] = compute_ece(
            np.array(all_confidences),
            np.array(all_accuracies),
        )
    else:
        results['ece']['Average'] = 0.0
    
    return results


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = 0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        n_in_bin = np.sum(in_bin)
        
        if n_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += n_in_bin * abs(avg_accuracy - avg_confidence)
            total_samples += n_in_bin
    
    if total_samples > 0:
        ece /= total_samples
    
    return ece


def print_results(results: Dict, title: str = "Results"):
    """Print results in a formatted table."""
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)
    
    metrics = ['accuracy', 'log_loss', 'ece']
    phases = [f'Phase_{i}' for i in range(1, NUM_PHASES + 1)]
    
    header = f"{'Phase':<12}"
    for metric in metrics:
        header += f"{metric.upper():<15}"
    print(header)
    print("-" * 90)
    
    for phase in phases:
        row = f"{phase:<12}"
        for metric in metrics:
            val = results[metric].get(phase, 0.0)
            row += f"{val:<15.4f}"
        print(row)
    
    print("-" * 90)
    avg_row = f"{'Average':<12}"
    for metric in metrics:
        val = results[metric].get('Average', 0.0)
        avg_row += f"{val:<15.4f}"
    print(avg_row)
    print("=" * 90)


def save_results(results: Dict, save_path: str):
    """Save results to CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    summary_data = []
    for metric in ['accuracy', 'log_loss', 'ece']:
        row = {'metric': metric}
        row.update(results[metric])
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="TabTransformer MSE Baseline for PGC")
    parser.add_argument("--train_folder", type=str, required=True,
                        help="Path to training data folder (all CSVs used)")
    parser.add_argument("--inference_folder", type=str, required=True,
                        help="Path to inference data folder for evaluation")
    parser.add_argument("--output_dir", type=str, default="results/baselines/tab_mse",
                        help="Output directory")
    parser.add_argument("--toy_ratio", type=float, default=1.0,
                        help="Ratio of training data to use (0.0-1.0)")
    
    # Model hyperparameters
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load training data
    print("\n" + "=" * 60)
    print("Loading Training Data")
    print("=" * 60)
    
    train_files = get_all_files_from_folder(args.train_folder, args.toy_ratio)
    train_loader, scaler_params = create_dataloader(
        folder_path=args.train_folder,
        file_list=train_files,
        scaler_params=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    
    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)
    
    input_dim = len(CONTINUOUS_FEATURES) + len(COMPUTED_FEATURES)
    print(f"Input dimension: {input_dim}")
    
    backbone = TransformerBackbone(
        input_dim=input_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    
    head = get_head("mse", embed_dim=args.embed_dim, dropout=args.dropout)
    
    backbone_params = sum(p.numel() for p in backbone.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    print(f"Backbone params: {backbone_params:,}")
    print(f"Head params: {head_params:,}")
    print(f"Total params: {backbone_params + head_params:,}")
    
    # Train
    print("\n" + "=" * 60)
    print("Training TabTransformer (MSE)")
    print("=" * 60)
    
    trainer = SimpleTrainer(
        backbone=backbone,
        head=head,
        train_loader=train_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )
    
    history = trainer.train(num_epochs=args.num_epochs)
    
    # Save checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    
    # Load inference data
    print("\n" + "=" * 60)
    print("Loading Inference Data")
    print("=" * 60)
    
    infer_files = get_all_files_from_folder(args.inference_folder, toy_ratio=1.0)
    infer_loader, _ = create_dataloader(
        folder_path=args.inference_folder,
        file_list=infer_files,
        scaler_params=scaler_params,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    
    print(f"Inference samples: {len(infer_loader.dataset)}")
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating on Inference Data")
    print("=" * 60)
    
    results = evaluate_on_inference(
        backbone=backbone,
        head=head,
        dataloader=infer_loader,
        device=device,
    )
    
    print_results(results, title="TabTransformer MSE Inference Results")
    
    # Save results
    save_results(results, os.path.join(args.output_dir, "inference_results.csv"))
    
    # Save scaler params
    scaler_path = os.path.join(args.output_dir, "scaler_params.json")
    scaler_to_save = {
        'mean': {k: float(v) for k, v in scaler_params['mean'].items()},
        'std': {k: float(v) for k, v in scaler_params['std'].items()},
    }
    with open(scaler_path, "w") as f:
        json.dump(scaler_to_save, f, indent=2)
    print(f"Scaler params saved to: {scaler_path}")
    
    # Save config
    config = {
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'input_dim': input_dim,
        'loss_type': 'mse',
        'train_folder': args.train_folder,
        'inference_folder': args.inference_folder,
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # Save history
    history_path = os.path.join(args.output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()


