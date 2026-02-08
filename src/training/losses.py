"""
Loss functions for PGC survival prediction.

This module implements advanced survival analysis loss functions tailored for
Winner Prediction in Battle Royale games.

Key Innovations for Top-Tier AI Conference:

1. CoxHazardLoss (Full Retrospective Hazard Learning - FullAllLoss):
   - Provides supervision at ALL time points prior to each squad's elimination
   - L_PGC_Full = -Σ_{k=1}^{p-1} [log_likelihood_eliminated + Σ log(1-hazard_prob)_surviving]
   - Unlike standard Cox PL which only supervises at elimination moments

2. RankCoxHazardLoss (Rank-Enhanced FullAllLoss - RankFullAllLoss):
   - Combines FullAllLoss with ranking penalty for temporal consistency
   - L_RankCox = L_PGC_Full + λ * Σ_{T_i>T_j} I[g_i>g_j] * exp((g_i-g_j)/σ)
   - Penalizes ranking violations where longer survivors have higher hazard

3. WeightedCoxHazardLoss (Time-Gap Weighted FullAllLoss - WeightedFullAllLoss):
   - FullAllLoss with time-gap weighting: (T_winner - T_i) / T_winner
   - L_Weighted = time_gap_weights * L_PGC_Full
   - Emphasizes squads eliminated early (clearer negative examples)

Conventions:
    - pred = hazard score g_θ (higher = more likely to be eliminated)
    - target = survival time T
    - Winner = squad with max(T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """
    Mean Squared Error loss for survival time regression.
    Predicts the actual survival time for each squad.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.criterion(pred, target)
        loss = loss * valid_mask.float()
        n_valid = valid_mask.sum()
        if n_valid > 0:
            return loss.sum() / n_valid
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class CoxHazardLoss(nn.Module):
    """
    Full Retrospective Hazard Learning Loss (FullAllLoss).
    
    Vectorized implementation without explicit loops.
    
    Mathematical formulation:
    L_PGC_Full = -Σ_{k=1}^{p-1} [
        log(exp(g_θ(x_{i*(t_k)}, t_k)) / Σ_{j∈R(t_k)} exp(g_θ(x_j, t_k))) +
        Σ_{i∈R(t_k)\i*(t_k)} log(1 - exp(g_θ(x_i, t_k)) / Σ_{j∈R(t_k)} exp(g_θ(x_j, t_k)))
    ]
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Hazard scores (B, N)
            target: Survival time (B, N)
            valid_mask: (B, N)
        """
        B, N = pred.shape
        device = pred.device
        
        # Mask invalid targets with -1 (so they don't affect max time logic)
        target_masked = torch.where(valid_mask, target, torch.tensor(-1.0, device=device))
        
        # 1. Expand dimensions for pairwise comparison (B, N, N)
        # T_i: time of squad i (potential elimination time)
        # T_j: time of squad j (potential risk set member)
        T_i = target_masked.unsqueeze(2)  # (B, N, 1)
        T_j = target_masked.unsqueeze(1)  # (B, 1, N)
        
        # 2. Identify Elimination Events
        # Winner (max time) is not an elimination event for this loss
        batch_max_time = target_masked.max(dim=1, keepdim=True)[0].unsqueeze(2) # (B, 1, 1)
        is_elimination = (T_i < batch_max_time) & (T_i > 0) & valid_mask.unsqueeze(2) # (B, N, 1)
        
        # 3. Identify Risk Sets R(t_i)
        # Squad j is in risk set of i if T_j >= T_i
        # Also need valid_mask for j
        risk_set_mask = (T_j >= T_i) & valid_mask.unsqueeze(1) # (B, N, N)
        
        # 4. Compute Log-Softmax Terms
        # We need sum_{j in R(t_i)} exp(g_j)
        # Use logsumexp for stability: log(sum(exp(x)))
        
        # Mask out non-risk-set members with -inf before logsumexp
        pred_j = pred.unsqueeze(1) # (B, 1, N)
        risk_pred_masked = pred_j.masked_fill(~risk_set_mask, float('-inf'))
        
        # log_denominator = log(sum_{j in R} exp(g_j))
        log_denominator = torch.logsumexp(risk_pred_masked, dim=2, keepdim=True) # (B, N, 1)
        
        # 5. Compute Probabilities
        # log_prob_j = g_j - log_denominator
        # This is log(P(elimination = j | risk set))
        log_probs = pred_j - log_denominator # (B, N, N)
        
        # 6. Part 1: Maximizing hazard of eliminated squad i*
        # Term: log(exp(g_i) / sum exp(g_j)) for i=i*
        # This corresponds to diagonal elements (j=i) where i is an elimination event
        log_prob_eliminated = torch.diagonal(log_probs, dim1=1, dim2=2).unsqueeze(2) # (B, N, 1)
        
        loss_part1 = -log_prob_eliminated * is_elimination # (B, N, 1)
        
        # 7. Part 2: Maximizing survival of others in risk set
        # Term: log(1 - exp(g_k) / sum exp(g_j)) for k in R(t_i) \ {i}
        # We need P(elimination = k | risk set)
        probs = torch.exp(log_probs) # (B, N, N)
        
        # Surviving mask: k in Risk Set AND k != i
        # Diagonal mask
        eye = torch.eye(N, device=device).unsqueeze(0).bool() # (1, N, N)
        surviving_mask = risk_set_mask & (~eye) # (B, N, N)
        
        # We want -log(1 - prob_k)
        # For numerical stability: 1 - exp(log_prob)
        # If prob is close to 1, loss explodes. Clamp for safety.
        probs_clamped = torch.clamp(probs, max=1.0 - self.eps)
        loss_surviving = -torch.log(1.0 - probs_clamped + self.eps)
        
        # Mask and sum over k
        loss_part2 = (loss_surviving * surviving_mask.float()).sum(dim=2, keepdim=True) # (B, N, 1)
        loss_part2 = loss_part2 * is_elimination
        
        # 8. Total Loss
        # Sum over all elimination events i
        total_loss_per_batch = (loss_part1 + loss_part2).sum(dim=1) # (B, 1)
        
        # Normalize by number of valid batches (or events) if needed
        # Current implementation sums, then averages over batch
        
        valid_batch_mask = (valid_mask.sum(dim=1) >= 2).float()
        final_loss = (total_loss_per_batch.squeeze() * valid_batch_mask).sum()
        num_valid = valid_batch_mask.sum().clamp(min=1.0)
        
        return final_loss / num_valid


class RankCoxHazardLoss(nn.Module):
    """
    Rank-Enhanced Full Retrospective Hazard Learning Loss (RankFullAllLoss).
    Vectorized implementation.
    """

    def __init__(self, eps: float = 1e-7, rank_weight: float = 1.0, sigma: float = 1.0):
        super().__init__()
        self.cox_loss = CoxHazardLoss(eps=eps)
        self.rank_weight = rank_weight
        self.sigma = sigma
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N = pred.shape
        device = pred.device
        
        # 1. Base FullAll Loss
        cox_loss = self.cox_loss(pred, target, valid_mask)
        
        # 2. Vectorized Ranking Penalty Term
        # Condition: T_i > T_j (i survived longer than j)
        T_i = target.unsqueeze(2) # (B, N, 1)
        T_j = target.unsqueeze(1) # (B, 1, N)
        time_diff = T_i - T_j
        
        valid_i = valid_mask.unsqueeze(2)
        valid_j = valid_mask.unsqueeze(1)
        valid_pair = valid_i & valid_j
        
        # Only consider pairs where i survived longer
        target_pair_mask = (time_diff > 0) & valid_pair
        
        # Condition: g_i > g_j (hazard of survivor > hazard of dead) - Violation!
        g_i = pred.unsqueeze(2)
        g_j = pred.unsqueeze(1)
        pred_diff = g_i - g_j
        
        violation_mask = (pred_diff > 0) & target_pair_mask
        
        # Penalty: exp((g_i - g_j) / sigma)
        penalties = torch.exp(pred_diff / self.sigma)
        
        # Sum penalties
        rank_loss_sum = (penalties * violation_mask.float()).sum(dim=(1, 2))
        
        # Normalize by number of valid pairs per batch
        num_pairs = target_pair_mask.float().sum(dim=(1, 2)).clamp(min=1.0)
        rank_loss = (rank_loss_sum / num_pairs).mean()
        
        return cox_loss + self.rank_weight * rank_loss


class WeightedCoxHazardLoss(nn.Module):
    """
    Time-Gap Weighted Full Retrospective Hazard Learning Loss (WeightedFullAllLoss).
    Vectorized implementation.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Hazard scores (B, N)
            target: Survival time (B, N)
            valid_mask: (B, N)
        """
        B, N = pred.shape
        device = pred.device
        
        target_masked = torch.where(valid_mask, target, torch.tensor(-1.0, device=device))
        
        # 1. Expand dimensions
        T_i = target_masked.unsqueeze(2) # (B, N, 1) - Elimination Time
        T_j = target_masked.unsqueeze(1) # (B, 1, N) - Risk Set Members
        
        # 2. Identify Elimination Events
        batch_max_time = target_masked.max(dim=1, keepdim=True)[0].unsqueeze(2)
        batch_max_time_safe = torch.clamp(batch_max_time, min=1.0)
        
        is_elimination = (T_i < batch_max_time) & (T_i > 0) & valid_mask.unsqueeze(2)
        
        # 3. Compute Time-Gap Weights
        # weight_i = (T_winner - T_i) / T_winner
        time_gap = batch_max_time - T_i
        weights = time_gap / batch_max_time_safe
        weights = torch.clamp(weights, min=0.0, max=1.0)
        
        # 4. Identify Risk Sets
        risk_set_mask = (T_j >= T_i) & valid_mask.unsqueeze(1)
        
        # 5. Compute Log-Softmax
        pred_j = pred.unsqueeze(1)
        risk_pred_masked = pred_j.masked_fill(~risk_set_mask, float('-inf'))
        log_denominator = torch.logsumexp(risk_pred_masked, dim=2, keepdim=True)
        log_probs = pred_j - log_denominator
        
        # 6. Part 1: Weighted eliminated squad loss
        log_prob_eliminated = torch.diagonal(log_probs, dim1=1, dim2=2).unsqueeze(2)
        loss_part1 = -log_prob_eliminated * is_elimination * weights
        
        # 7. Part 2: Weighted surviving squads loss
        probs = torch.exp(log_probs)
        eye = torch.eye(N, device=device).unsqueeze(0).bool()
        surviving_mask = risk_set_mask & (~eye)
        
        probs_clamped = torch.clamp(probs, max=1.0 - self.eps)
        loss_surviving = -torch.log(1.0 - probs_clamped + self.eps)
        
        # Sum over k, then multiply by weight of time point i
        loss_part2_sum = (loss_surviving * surviving_mask.float()).sum(dim=2, keepdim=True)
        loss_part2 = loss_part2_sum * is_elimination * weights
        
        # 8. Total Loss
        total_loss_per_batch = (loss_part1 + loss_part2).sum(dim=1)
        
        valid_batch_mask = (valid_mask.sum(dim=1) >= 2).float()
        final_loss = (total_loss_per_batch.squeeze() * valid_batch_mask).sum()
        num_valid = valid_batch_mask.sum().clamp(min=1.0)
        
        return final_loss / num_valid


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for winner prediction.
    Standard classification loss treating the squad with max survival time as the class.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N = pred.shape
        device = pred.device

        target_masked = torch.where(
            valid_mask,
            target,
            torch.tensor(float('-inf'), device=device, dtype=target.dtype)
        )
        winner_indices = target_masked.argmax(dim=-1)
        
        max_survival = target_masked.max(dim=-1)[0]
        valid_batch_mask = max_survival > float('-inf')
        
        if not valid_batch_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        LARGE_NEG = -1e9
        pred_masked = torch.where(
            valid_mask,
            pred,
            torch.tensor(LARGE_NEG, device=device, dtype=pred.dtype)
        )

        log_probs = F.log_softmax(pred_masked, dim=-1)
        loss = F.nll_loss(log_probs, winner_indices, reduction='none')
        loss = loss * valid_batch_mask.float()
        
        return loss.sum() / (valid_batch_mask.sum() + 1e-7)


class ConcordanceLoss(nn.Module):
    """
    Concordance-based Loss (C-index) for Winner with Time-Gap Weighting.
    
    Enhanced to weight pairwise comparisons by survival time difference.
    """
    def __init__(self, margin: float = 0.0, time_weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.time_weight = time_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N = pred.shape
        device = pred.device
        
        target_masked = torch.where(valid_mask, target, torch.tensor(float('-inf'), device=device))
        winner = torch.argmax(target_masked, dim=1)
        
        valid_batch_mask = target_masked.max(dim=1)[0] > float('-inf')
        if not valid_batch_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Winner's score and time
        pred_winner = pred.gather(1, winner.unsqueeze(1)).squeeze(1)  # (B,)
        time_winner = target.gather(1, winner.unsqueeze(1))  # (B, 1)
        
        # Score difference
        score_diff = pred_winner.unsqueeze(1) - pred  # (B, N)
        
        # Time gap weight: (T_winner - T_j) / T_winner
        time_gap = time_winner - target  # (B, N)
        time_gap_norm = time_gap / (time_winner + 1e-7)
        time_gap_norm = torch.clamp(time_gap_norm, min=0.0, max=1.0)
        
        # Combined weight: 1 + time_weight * time_gap_norm
        pair_weight = 1.0 + self.time_weight * time_gap_norm
        
        # Mask for non-winner valid squads
        winner_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        winner_mask.scatter_(1, winner.unsqueeze(1), True)
        other_mask = valid_mask & ~winner_mask
        
        # Weighted log-sigmoid loss
        score_diff_with_margin = score_diff - self.margin
        log_sigmoid = F.logsigmoid(score_diff_with_margin)
        
        weighted_loss = -log_sigmoid * pair_weight * other_mask.float() * valid_batch_mask.unsqueeze(1).float()
        
        total_loss = weighted_loss.sum()
        total_weight = (pair_weight * other_mask.float() * valid_batch_mask.unsqueeze(1).float()).sum()
        
        if total_weight > 0:
            return total_loss / total_weight
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class SurvivalCELoss(nn.Module):
    """
    Survival Score Cross-Entropy Loss.
    Mathematically equivalent to CrossEntropyLoss but explicit in intent.
    """
    def __init__(self):
        super().__init__()
        self.ce = CrossEntropyLoss()

    def forward(self, pred, target, valid_mask):
        return self.ce(pred, target, valid_mask)


def get_loss_fn(loss_type: str) -> nn.Module:
    loss_type = loss_type.lower()

    if loss_type == 'mse':
        return MSELoss()
    elif loss_type == 'cox':
        return CoxHazardLoss(eps=1e-7)
    elif loss_type in ('ce', 'cross_entropy', 'crossentropy'):
        return CrossEntropyLoss()
    elif loss_type in ('rank_cox', 'rankcox', 'listmle'):
        return RankCoxHazardLoss(eps=1e-7, rank_weight=1.0, sigma=1.0)
    elif loss_type in ('weighted_cox', 'weightedcox'):
        return WeightedCoxHazardLoss(eps=1e-7)
    elif loss_type in ('concordance', 'cindex', 'c_index'):
        return ConcordanceLoss(time_weight=1.0)
    elif loss_type in ('survival_ce', 'survivalce', 'sce'):
        return SurvivalCELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
