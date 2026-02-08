"""
Reusable modules for PGC models.
"""

import math
import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    """
    Fourier-based positional encoding for continuous coordinates.

    Args:
        coord_dim: Dimension of input coordinates (e.g., 3 for xyz).
        embed_dim: Output embedding dimension.
        num_frequencies: Number of frequency bands.
    """

    def __init__(
        self,
        coord_dim: int,
        embed_dim: int,
        num_frequencies: int = 8,
    ):
        super().__init__()

        self.coord_dim = coord_dim
        self.num_frequencies = num_frequencies

        # Fourier features: coord_dim * num_frequencies * 2 (sin + cos)
        fourier_dim = coord_dim * num_frequencies * 2

        # Project to embed_dim
        self.proj = nn.Linear(fourier_dim, embed_dim)

        # Frequency bands (learnable or fixed)
        frequencies = 2.0 ** torch.arange(num_frequencies).float()
        self.register_buffer('frequencies', frequencies)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (..., coord_dim) - input coordinates

        Returns:
            (..., embed_dim) - positional embeddings
        """
        # coords: (..., coord_dim)
        # Expand frequencies: (num_frequencies,)
        # Result: (..., coord_dim, num_frequencies)
        scaled = coords.unsqueeze(-1) * self.frequencies * math.pi

        # Fourier features: (..., coord_dim, num_frequencies * 2)
        fourier = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)

        # Flatten: (..., coord_dim * num_frequencies * 2)
        fourier = fourier.flatten(-2)

        # Project: (..., embed_dim)
        return self.proj(fourier)


class ZoneEmbedding(nn.Module):
    """
    Embedding for zone information (x, y, radius).

    Treats (x, y) as 2D position and radius separately.

    Args:
        embed_dim: Output embedding dimension.
        num_frequencies: Number of frequency bands for positional encoding.
    """

    def __init__(
        self,
        embed_dim: int,
        num_frequencies: int = 8,
    ):
        super().__init__()

        # 2D positional encoding for (x, y)
        self.pos_encoding = FourierPositionalEncoding(
            coord_dim=2,
            embed_dim=embed_dim // 2,
            num_frequencies=num_frequencies,
        )

        # MLP for radius
        self.radius_mlp = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim // 2),
        )

        # Final projection
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, zone_info: torch.Tensor) -> torch.Tensor:
        """
        Args:
            zone_info: (batch, 3) - [x, y, radius]

        Returns:
            (batch, embed_dim) - zone embedding
        """
        # Split into position and radius
        xy = zone_info[:, :2]  # (batch, 2)
        radius = zone_info[:, 2:3]  # (batch, 1)

        # Encode position and radius
        pos_emb = self.pos_encoding(xy)  # (batch, embed_dim // 2)
        radius_emb = self.radius_mlp(radius)  # (batch, embed_dim // 2)

        # Concatenate and project
        combined = torch.cat([pos_emb, radius_emb], dim=-1)
        return self.proj(combined)


class TokenEmbedding(nn.Module):
    """
    MLP-based token embedding for continuous features.

    Args:
        input_dim: Number of input features.
        embed_dim: Dimension of token embedding.
        hidden_dim: Hidden dimension of MLP. If None, uses embed_dim.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_squads, input_dim)

        Returns:
            (batch, num_squads, embed_dim)
        """
        return self.mlp(x)

