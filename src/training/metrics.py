"""
Evaluation metrics for PGC models.

Includes:
- Accuracy: Whether predicted winner matches actual winner
- ECE (Expected Calibration Error): Calibration of probability predictions
- Log Loss: Cross-entropy loss for probability predictions
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_winner_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """
    Compute accuracy for winner prediction.

    Args:
        pred: Predictions (batch, num_squads)
        target: Targets (batch, num_squads)
        valid_mask: (batch, num_squads), True if valid

    Returns:
        Accuracy (0.0 to 1.0)
    """
    # Mask out invalid squads
    pred_masked = pred.clone()
    target_masked = target.clone()
    pred_masked[~valid_mask] = float('-inf')
    target_masked[~valid_mask] = float('-inf')

    # Get winners
    pred_winners = pred_masked.argmax(dim=-1)
    target_winners = target_masked.argmax(dim=-1)

    # Compute accuracy
    correct = (pred_winners == target_winners).float()
    return correct.mean().item()


def compute_winner_probabilities(
    pred: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert predictions to winner probabilities using softmax.

    Args:
        pred: Predictions (batch, num_squads)
        valid_mask: (batch, num_squads), True if valid

    Returns:
        Probabilities (batch, num_squads) with invalid positions set to 0
    """
    # Set invalid positions to -inf for softmax
    pred_masked = pred.clone()
    pred_masked[~valid_mask] = float('-inf')

    # Apply softmax to get probabilities
    probs = F.softmax(pred_masked, dim=-1)

    # Set invalid positions to 0
    probs[~valid_mask] = 0.0

    return probs


def compute_log_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-7,
) -> float:
    """
    Compute log loss (cross-entropy) for winner prediction.

    Args:
        pred: Predictions (batch, num_squads)
        target: Targets (batch, num_squads)
        valid_mask: (batch, num_squads), True if valid
        eps: Small value to avoid log(0)

    Returns:
        Log loss value
    """
    batch_size = pred.size(0)

    # Get winner probabilities
    probs = compute_winner_probabilities(pred, valid_mask)

    # Get target winners (one-hot)
    target_masked = target.clone()
    target_masked[~valid_mask] = float('-inf')
    target_winners = target_masked.argmax(dim=-1)  # (batch,)

    # Get probability assigned to true winner
    winner_probs = probs[torch.arange(batch_size), target_winners]

    # Clamp for numerical stability
    winner_probs = torch.clamp(winner_probs, min=eps, max=1.0 - eps)

    # Compute log loss
    log_loss = -torch.log(winner_probs).mean().item()

    return log_loss


def compute_ece(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how well the predicted probabilities match the actual
    accuracy. A well-calibrated model should have ECE close to 0.

    Args:
        pred: Predictions (batch, num_squads)
        target: Targets (batch, num_squads)
        valid_mask: (batch, num_squads), True if valid
        n_bins: Number of bins for calibration

    Returns:
        ECE value (0.0 to 1.0)
    """
    batch_size = pred.size(0)

    # Get winner probabilities
    probs = compute_winner_probabilities(pred, valid_mask)

    # Get target winners
    target_masked = target.clone()
    target_masked[~valid_mask] = float('-inf')
    target_winners = target_masked.argmax(dim=-1)

    # Get predicted winners and their confidence
    pred_winners = probs.argmax(dim=-1)
    confidences = probs.max(dim=-1).values  # (batch,)

    # Check if predictions are correct
    correct = (pred_winners == target_winners).float()

    # Bin confidences and compute ECE
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = 0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        n_in_bin = in_bin.sum().item()

        if n_in_bin > 0:
            # Compute average confidence and accuracy in bin
            avg_confidence = confidences[in_bin].mean().item()
            avg_accuracy = correct[in_bin].mean().item()

            # Add weighted absolute difference to ECE
            ece += n_in_bin * abs(avg_accuracy - avg_confidence)
            total_samples += n_in_bin

    if total_samples > 0:
        ece /= total_samples

    return ece


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        pred: Predictions (batch, num_squads)
        target: Targets (batch, num_squads)
        valid_mask: (batch, num_squads), True if valid

    Returns:
        Dictionary with accuracy, log_loss, and ece.
    """
    return {
        'accuracy': compute_winner_accuracy(pred, target, valid_mask),
        'log_loss': compute_log_loss(pred, target, valid_mask),
        'ece': compute_ece(pred, target, valid_mask),
    }


