"""
PGC Dataset and DataLoader for PyTorch.
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data.continuous_features import CONTINUOUS_FEATURES
from src.data.utils import parse_location_features


LOCATION_FEATURES = ['positions', 'bluezone_info', 'whitezone_info']

# Scaling constants
POSITION_SCALE = 80000.0
ZONE_SCALE = 800000.0
TARGET_SCALE = 2000.0

# Maximum number of squads (for padding)
MAX_SQUADS = 16

# Map types for embedding
MAP_TYPES = ['erangel', 'miramar', 'taego', 'rondo']
MAP_TO_IDX = {map_name: idx for idx, map_name in enumerate(MAP_TYPES)}
NUM_MAPS = len(MAP_TYPES)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable number of squads.

    Pads all samples to MAX_SQUADS and creates a padding mask.

    Args:
        batch: List of sample dictionaries from PGCDataset.

    Returns:
        Batched dictionary with padded tensors and masks.
    """
    batch_size = len(batch)

    # Get dimensions from first sample
    num_features = batch[0]['x_continuous'].shape[1]

    # Initialize padded tensors
    x_continuous = torch.zeros(batch_size, MAX_SQUADS, num_features)
    positions = torch.full((batch_size, MAX_SQUADS, 4, 3), float('nan'))
    bluezone_info = torch.zeros(batch_size, 3)
    whitezone_info = torch.zeros(batch_size, 3)
    y = torch.zeros(batch_size, MAX_SQUADS)
    squad_alive_mask = torch.zeros(batch_size, MAX_SQUADS, dtype=torch.bool)
    padding_mask = torch.zeros(batch_size, MAX_SQUADS, dtype=torch.bool)
    num_squads = torch.zeros(batch_size, dtype=torch.long)
    time_point = torch.zeros(batch_size)
    map_idx = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(batch):
        n_squads = sample['num_squads']
        num_squads[i] = n_squads

        # Copy data (up to MAX_SQUADS)
        actual_squads = min(n_squads, MAX_SQUADS)

        x_continuous[i, :actual_squads] = sample['x_continuous'][:actual_squads]
        positions[i, :actual_squads] = sample['positions'][:actual_squads]
        bluezone_info[i] = sample['bluezone_info']
        whitezone_info[i] = sample['whitezone_info']
        y[i, :actual_squads] = sample['y'][:actual_squads]
        squad_alive_mask[i, :actual_squads] = sample['squad_alive_mask'][:actual_squads]
        padding_mask[i, :actual_squads] = True  # True for valid (non-padded) positions
        time_point[i] = sample['time_point']
        map_idx[i] = sample['map_idx']

    return {
        'x_continuous': x_continuous,      # (batch, MAX_SQUADS, num_features)
        'positions': positions,            # (batch, MAX_SQUADS, 4, 3)
        'bluezone_info': bluezone_info,    # (batch, 3)
        'whitezone_info': whitezone_info,  # (batch, 3)
        'y': y,                            # (batch, MAX_SQUADS)
        'squad_alive_mask': squad_alive_mask,  # (batch, MAX_SQUADS)
        'padding_mask': padding_mask,      # (batch, MAX_SQUADS) True if valid
        'num_squads': num_squads,          # (batch,)
        'time_point': time_point,          # (batch,)
        'map_idx': map_idx,                # (batch,)
    }


class PGCDataset(Dataset):
    """
    PyTorch Dataset for PGC match data.

    Each sample consists of all squads from the same (match_id, time_point).
    This forms a sequence of squad tokens.

    Args:
        folder_path: Path to folder containing CSV files.
        file_list: List of CSV filenames to load.
        continuous_features: List of continuous feature column names to use.
        target_col: Name of the target column. Default is "squad_death_time".
        scaler_params: Tuple of (mean, std) for standardization.
                       If None, no scaling is applied.
    """

    def __init__(
        self,
        folder_path: str,
        file_list: List[str],
        continuous_features: List[str] = CONTINUOUS_FEATURES,
        target_col: str = "squad_death_time",
        scaler_params: Tuple[np.ndarray, np.ndarray] = None,
    ):
        self.folder_path = folder_path
        self.file_list = file_list
        self.continuous_features = continuous_features
        self.target_col = target_col
        self.scaler_params = scaler_params

        self.data = self._load_data()
        self.group_keys, self.group_row_indices = self._build_group_index()

        if self.scaler_params is not None:
            self._apply_scaling()

    def _load_data(self) -> pd.DataFrame:
        """Load and concatenate CSV files from the file list."""
        if not self.file_list:
            raise ValueError("file_list is empty")

        dfs = []
        for csv_file in tqdm(self.file_list, desc="Loading CSV files"):
            file_path = os.path.join(self.folder_path, csv_file)
            df = pd.read_csv(file_path)
            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)

        # Validate columns exist
        missing_features = [
            f for f in self.continuous_features
            if f not in data.columns
        ]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")

        if self.target_col not in data.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in data"
            )

        # Validate location features exist
        missing_location = [
            f for f in LOCATION_FEATURES
            if f not in data.columns
        ]
        if missing_location:
            raise ValueError(
                f"Missing location features in data: {missing_location}"
            )

        # Validate grouping columns exist
        for col in ['match_id', 'time_point']:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        return data

    def _build_group_index(self) -> Tuple[List[Tuple[str, float]], List[List[int]]]:
        """
        Build index of unique (match_id, time_point) combinations and row indices.

        Returns:
            Tuple of:
            - List of (match_id, time_point) keys
            - List of row indices for each group
        """
        grouped = self.data.groupby(['match_id', 'time_point'])
        group_keys = list(grouped.groups.keys())
        # Store actual row indices for each group to avoid float comparison issues
        group_row_indices = [indices.tolist() for indices in grouped.groups.values()]
        return group_keys, group_row_indices

    def _apply_scaling(self):
        """Apply standard scaling using provided scaler parameters."""
        mean, std = self.scaler_params
        self.data[self.continuous_features] = (
            self.data[self.continuous_features] - mean
        ) / std

    def compute_scaler_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std from the data for standardization."""
        mean = self.data[self.continuous_features].mean().values
        std = self.data[self.continuous_features].std().values
        # Avoid division by zero
        std[std == 0] = 1.0
        return mean, std

    def __len__(self) -> int:
        return len(self.group_keys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get row indices for this group (avoids float comparison issues)
        row_indices = self.group_row_indices[idx]
        group_data = self.data.iloc[row_indices]

        num_squads = len(group_data)
        time_point = group_data.iloc[0]['time_point']

        # Extract continuous features: (num_squads, num_features)
        x_continuous = torch.tensor(
            group_data[self.continuous_features].values.astype(float),
            dtype=torch.float32
        )

        # Extract and parse location features
        positions_list = []
        for _, row in group_data.iterrows():
            pos, _, _ = parse_location_features(
                positions_str=row['positions'],
                bluezone_str=row['bluezone_info'],
                whitezone_str=row['whitezone_info'],
            )
            positions_list.append(pos)

        # positions: (num_squads, 4, 3), scaled by POSITION_SCALE
        positions = torch.tensor(
            np.stack(positions_list, axis=0),
            dtype=torch.float32
        ) / POSITION_SCALE

        # squad_alive_mask: True if squad is still alive
        # A squad is eliminated if:
        #   1. ALL 4 players have nan coordinates, OR
        #   2. squad_total_health is 0
        # positions shape: (num_squads, 4, 3)
        positions_valid = ~torch.isnan(positions).all(dim=-1).all(dim=-1)
        
        # Check squad_total_health (0 means eliminated)
        squad_health = torch.tensor(
            group_data['squad_total_health'].values.astype(float),
            dtype=torch.float32
        )
        health_valid = squad_health > 0
        
        # Squad is alive only if both conditions are satisfied
        squad_alive_mask = positions_valid & health_valid

        # bluezone_info and whitezone_info are same for all squads in a group
        # scaled by ZONE_SCALE
        first_row = group_data.iloc[0]
        _, bluezone_info, whitezone_info = parse_location_features(
            positions_str=first_row['positions'],
            bluezone_str=first_row['bluezone_info'],
            whitezone_str=first_row['whitezone_info'],
        )
        bluezone_info = torch.tensor(
            bluezone_info, dtype=torch.float32
        ) / ZONE_SCALE
        whitezone_info = torch.tensor(
            whitezone_info, dtype=torch.float32
        ) / ZONE_SCALE

        # Extract target: (num_squads,), scaled by TARGET_SCALE
        y = torch.tensor(
            group_data[self.target_col].values.astype(float),
            dtype=torch.float32
        ) / TARGET_SCALE

        # Extract map type (same for all squads in a match)
        map_name = first_row['map']
        map_idx = MAP_TO_IDX.get(map_name, 0)  # Default to 0 if unknown

        return {
            'x_continuous': x_continuous,      # (num_squads, num_features)
            'positions': positions,            # (num_squads, 4, 3)
            'bluezone_info': bluezone_info,    # (3,)
            'whitezone_info': whitezone_info,  # (3,)
            'y': y,                            # (num_squads,)
            'squad_alive_mask': squad_alive_mask,  # (num_squads,) True if alive
            'num_squads': num_squads,
            'time_point': torch.tensor(time_point, dtype=torch.float32),
            'map_idx': torch.tensor(map_idx, dtype=torch.long),  # scalar
        }


def get_dataloaders(
    folder_path: str,
    split_csv_path: str,
    continuous_features: List[str] = CONTINUOUS_FEATURES,
    target_col: str = "squad_death_time",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test DataLoaders with standard scaling.

    Args:
        folder_path: Path to folder containing CSV files.
        split_csv_path: Path to CSV file containing split information.
                        Expected columns: 'filename', 'split'.
        continuous_features: List of continuous feature column names to use.
        target_col: Name of the target column. Default is "squad_death_time".
        batch_size: Batch size for DataLoader.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Load split information
    split_df = pd.read_csv(split_csv_path)

    train_files = split_df[split_df['split'] == 'train']['filename'].tolist()
    val_files = split_df[split_df['split'] == 'val']['filename'].tolist()
    test_files = split_df[split_df['split'] == 'test']['filename'].tolist()

    # Create train dataset (without scaling first to compute scaler params)
    train_dataset = PGCDataset(
        folder_path=folder_path,
        file_list=train_files,
        continuous_features=continuous_features,
        target_col=target_col,
        scaler_params=None,
    )

    # Compute scaler parameters from train data
    scaler_params = train_dataset.compute_scaler_params()

    # Apply scaling to train dataset
    train_dataset.scaler_params = scaler_params
    train_dataset._apply_scaling()

    # Create val dataset with train scaler params
    val_dataset = PGCDataset(
        folder_path=folder_path,
        file_list=val_files,
        continuous_features=continuous_features,
        target_col=target_col,
        scaler_params=scaler_params,
    )

    # Create test dataset with train scaler params
    test_dataset = PGCDataset(
        folder_path=folder_path,
        file_list=test_files,
        continuous_features=continuous_features,
        target_col=target_col,
        scaler_params=scaler_params,
    )

    # Create DataLoaders with custom collate_fn for variable squad sizes
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    folder_path = "/path/to/csv/folder"
    split_csv_path = "/path/to/split.csv"

    train_loader, val_loader, test_loader = get_dataloaders(
        folder_path=folder_path,
        split_csv_path=split_csv_path,
        batch_size=1,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    for batch in train_loader:
        print(f"x_continuous shape: {batch['x_continuous'].shape}")
        print(f"positions shape: {batch['positions'].shape}")
        print(f"bluezone_info shape: {batch['bluezone_info'].shape}")
        print(f"whitezone_info shape: {batch['whitezone_info'].shape}")
        print(f"y shape: {batch['y'].shape}")
        print(f"num_squads: {batch['num_squads']}")
        break

