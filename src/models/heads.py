"""
Prediction heads for PGC models.

Head types:
- RegressionHead: For MSE loss (survival time prediction)
- HazardHead: For Cox loss (log-hazard prediction)
- ClassificationHead: For Cross-entropy loss (winner classification)
"""

import torch
import torch.nn as nn


class SurvivalTimeHead(nn.Module):
    """
    Prediction head for squad survival time (regression).

    Used with MSE loss.
    Output: Predicted survival time (higher = survives longer).

    Args:
        embed_dim: Dimension of input embeddings.
        hidden_dim: Hidden dimension of MLP. If None, uses embed_dim.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_squads, embed_dim) - squad embeddings

        Returns:
            (batch, num_squads) - predicted survival time per squad
        """
        return self.mlp(x).squeeze(-1)


class HazardHead(nn.Module):
    """
    Prediction head for log-hazard (Cox survival analysis).

    Used with Cox Hazard loss.
    Output: Log-hazard (higher = more likely to die soon).

    Args:
        embed_dim: Dimension of input embeddings.
        hidden_dim: Hidden dimension of MLP. If None, uses embed_dim.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_squads, embed_dim) - squad embeddings

        Returns:
            (batch, num_squads) - log-hazard per squad
        """
        return self.mlp(x).squeeze(-1)


class ClassificationHead(nn.Module):
    """
    Prediction head for winner classification.

    Used with Cross-entropy loss.
    Output: Logits for each squad (used with softmax for probability).

    Args:
        embed_dim: Dimension of input embeddings.
        hidden_dim: Hidden dimension of MLP. If None, uses embed_dim.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_squads, embed_dim) - squad embeddings

        Returns:
            (batch, num_squads) - logits per squad
        """
        return self.mlp(x).squeeze(-1)


def get_head(
    loss_type: str,
    embed_dim: int,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Get prediction head based on loss type.

    Args:
        loss_type: One of 'mse', 'cox', 'ce'
        embed_dim: Embedding dimension.
        dropout: Dropout probability.

    Returns:
        Prediction head module.
    """
    loss_type = loss_type.lower()

    if loss_type == 'mse':
        return SurvivalTimeHead(embed_dim=embed_dim, dropout=dropout)
    elif loss_type in ('cox', 'rank_cox', 'rankcox', 'listmle', 'weighted_cox', 'weightedcox'):
        return HazardHead(embed_dim=embed_dim, dropout=dropout)
    elif loss_type in ('ce', 'cross_entropy', 'crossentropy', 
                       'concordance', 'cindex', 'c_index',
                       'survival_ce', 'survivalce', 'sce'):
        # All classification-style losses use ClassificationHead
        # (outputs logits/scores for each squad)
        return ClassificationHead(embed_dim=embed_dim, dropout=dropout)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Choose from: 'mse', 'cox', 'ce', 'rank_cox', 'weighted_cox', "
            f"'concordance', 'survival_ce'"
        )

