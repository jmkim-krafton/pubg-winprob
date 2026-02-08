"""
LightGBM Baseline for PGC Survival Prediction.

Uses the same train/val/test split as the Transformer model.
Trains on individual squad rows, but evaluates by grouping
(match_id, time_point) for fair comparison.
"""

import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.continuous_features import CONTINUOUS_FEATURES
from src.data.utils import parse_positions, parse_zone_info


# Constants (same as Transformer)
TARGET_SCALE = 2000.0
NUM_PHASES = 10
SAMPLES_PER_PHASE = 5

# Position feature names
POSITION_FEATURES = [
    'team_center_x', 'team_center_y', 'team_center_z',
    'team_std_x', 'team_std_y', 'team_std_z',
    'bluezone_x', 'bluezone_y',
    'whitezone_x', 'whitezone_y',
]

# Distance features (same as dataset_v2)
DISTANCE_FEATURES = [
    'dist_from_bluezone_v2',
    'dist_from_whitezone_v2',
]

# Scaling constants
POSITION_SCALE = 80000.0
ZONE_SCALE = 800000.0


def extract_position_features(
    df: pd.DataFrame,
    use_dataset_v2: bool = True,
) -> pd.DataFrame:
    """
    Extract position-related features from positions, bluezone_info, whitezone_info.
    
    Features:
    - team_center_x/y/z: Mean position of alive players
    - team_std_x/y/z: Std of alive player positions
    - bluezone_x/y: Bluezone center coordinates
    - whitezone_x/y: Whitezone center coordinates
    
    If use_dataset_v2=True, also computes:
    - dist_from_bluezone_v2: distance from team center to bluezone / radius
    - dist_from_whitezone_v2: distance from team center to whitezone / radius
    """
    # Initialize feature columns
    for col in POSITION_FEATURES:
        df[col] = 0.0
    
    # Initialize distance features if using v2
    if use_dataset_v2:
        for col in DISTANCE_FEATURES:
            df[col] = 0.0
    
    for idx in tqdm(df.index, desc="Extracting position features"):
        row = df.loc[idx]
        
        team_center_xy = None  # Store for distance calculation
        
        # Parse positions (4 players, 3 coords each)
        try:
            positions = parse_positions(row['positions'])  # (4, 3)
            
            # Get alive player positions (non-nan)
            alive_mask = ~np.isnan(positions[:, 0])
            alive_positions = positions[alive_mask]
            
            if len(alive_positions) > 0:
                # Team center (mean of alive players)
                center = np.nanmean(alive_positions, axis=0)
                team_center_xy = center[:2]  # x, y for distance calc
                
                center_scaled = center / POSITION_SCALE
                df.loc[idx, 'team_center_x'] = center_scaled[0]
                df.loc[idx, 'team_center_y'] = center_scaled[1]
                df.loc[idx, 'team_center_z'] = center_scaled[2]
                
                # Team spread (std of alive players)
                if len(alive_positions) > 1:
                    std = np.nanstd(alive_positions, axis=0) / POSITION_SCALE
                    df.loc[idx, 'team_std_x'] = std[0]
                    df.loc[idx, 'team_std_y'] = std[1]
                    df.loc[idx, 'team_std_z'] = std[2]
        except Exception:
            pass
        
        bluezone = None
        whitezone = None
        
        # Parse bluezone info
        try:
            bluezone = parse_zone_info(row['bluezone_info'])  # (3,) = x, y, radius
            df.loc[idx, 'bluezone_x'] = bluezone[0] / ZONE_SCALE
            df.loc[idx, 'bluezone_y'] = bluezone[1] / ZONE_SCALE
        except Exception:
            pass
        
        # Parse whitezone info
        try:
            whitezone = parse_zone_info(row['whitezone_info'])  # (3,) = x, y, radius
            df.loc[idx, 'whitezone_x'] = whitezone[0] / ZONE_SCALE
            df.loc[idx, 'whitezone_y'] = whitezone[1] / ZONE_SCALE
        except Exception:
            pass
        
        # Compute distance features if using v2
        if use_dataset_v2 and team_center_xy is not None:
            # Distance to bluezone center / radius
            if bluezone is not None and bluezone[2] > 0:
                dist_blue = np.sqrt(
                    (team_center_xy[0] - bluezone[0]) ** 2 +
                    (team_center_xy[1] - bluezone[1]) ** 2
                )
                df.loc[idx, 'dist_from_bluezone_v2'] = dist_blue / bluezone[2]
            
            # Distance to whitezone center / radius
            if whitezone is not None and whitezone[2] > 0:
                dist_white = np.sqrt(
                    (team_center_xy[0] - whitezone[0]) ** 2 +
                    (team_center_xy[1] - whitezone[1]) ** 2
                )
                df.loc[idx, 'dist_from_whitezone_v2'] = dist_white / whitezone[2]
    
    return df


def load_data(
    folder_path: str,
    split_csv_path: str,
    toy_ratio: float = 1.0,
    use_dataset_v2: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, List[str]]:
    """
    Load train/val/test data using the same split as Transformer.

    Args:
        folder_path: Path to data folder.
        split_csv_path: Path to split CSV.
        toy_ratio: Ratio of data to use.
        use_dataset_v2: If True, include distance features.

    Returns:
        Tuple of (train_df, val_df, test_df, scaler_params, all_features)
    """
    import random

    # Load split info
    split_df = pd.read_csv(split_csv_path)
    train_files = split_df[split_df['split'] == 'train']['filename'].tolist()
    val_files = split_df[split_df['split'] == 'val']['filename'].tolist()
    test_files = split_df[split_df['split'] == 'test']['filename'].tolist()

    # Apply toy ratio
    if toy_ratio < 1.0:
        print(f"Toy mode: using {toy_ratio*100:.0f}% of data")
        random.seed(42)
        train_files = random.sample(train_files, max(1, int(len(train_files) * toy_ratio)))
        val_files = random.sample(val_files, max(1, int(len(val_files) * toy_ratio)))
        test_files = random.sample(test_files, max(1, int(len(test_files) * toy_ratio)))
        print(f"  Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    def load_csvs(file_list: List[str]) -> pd.DataFrame:
        dfs = []
        for f in tqdm(file_list, desc="Loading"):
            df = pd.read_csv(os.path.join(folder_path, f))
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    print("Loading train data...")
    train_df = load_csvs(train_files)
    print("Loading val data...")
    val_df = load_csvs(val_files)
    print("Loading test data...")
    test_df = load_csvs(test_files)

    # Extract position features (and distance features if v2)
    print(f"Extracting position features (use_dataset_v2={use_dataset_v2})...")
    print("  Processing train...")
    train_df = extract_position_features(train_df, use_dataset_v2=use_dataset_v2)
    print("  Processing val...")
    val_df = extract_position_features(val_df, use_dataset_v2=use_dataset_v2)
    print("  Processing test...")
    test_df = extract_position_features(test_df, use_dataset_v2=use_dataset_v2)

    # All features = continuous + position (+ distance if v2)
    if use_dataset_v2:
        all_features = CONTINUOUS_FEATURES + POSITION_FEATURES + DISTANCE_FEATURES
    else:
        all_features = CONTINUOUS_FEATURES + POSITION_FEATURES

    # Compute scaler params from train (continuous features only)
    mean = train_df[CONTINUOUS_FEATURES].mean()
    std = train_df[CONTINUOUS_FEATURES].std()
    std[std == 0] = 1.0
    scaler_params = {'mean': mean, 'std': std}

    # Apply scaling to continuous features
    for df in [train_df, val_df, test_df]:
        df[CONTINUOUS_FEATURES] = (df[CONTINUOUS_FEATURES] - mean) / std

    print(f"Train: {len(train_df):,} rows")
    print(f"Val: {len(val_df):,} rows")
    print(f"Test: {len(test_df):,} rows")
    n_pos = len(POSITION_FEATURES)
    n_dist = len(DISTANCE_FEATURES) if use_dataset_v2 else 0
    print(f"Features: {len(all_features)} (continuous: {len(CONTINUOUS_FEATURES)}, "
          f"position: {n_pos}, distance: {n_dist})")

    return train_df, val_df, test_df, scaler_params, all_features


def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    all_features: List[str],
    params: Dict = None,
) -> lgb.Booster:
    """
    Train LightGBM model for survival time prediction.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        all_features: List of feature column names to use.
        params: LightGBM parameters.

    Returns:
        Trained LightGBM Booster.
    """
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
        }

    X_train = train_df[all_features]
    y_train = train_df['squad_death_time'] / TARGET_SCALE

    X_val = val_df[all_features]
    y_val = val_df['squad_death_time'] / TARGET_SCALE

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    print("Training LightGBM...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    print(f"Best iteration: {model.best_iteration}")

    return model


def compute_metrics_by_phase(
    df: pd.DataFrame,
    pred_col: str = 'pred',
    target_col: str = 'squad_death_time',
) -> Dict:
    """
    Compute phase-wise metrics (accuracy, log_loss, ece).

    Groups by (match_id, time_point) and computes winner prediction metrics.
    """
    from scipy.special import softmax

    # Group by match_id and sort by time_point
    match_groups = df.groupby('match_id')

    # Build phase mapping
    phase_data = defaultdict(list)

    for match_id, match_df in match_groups:
        # Sort by time_point
        time_points = sorted(match_df['time_point'].unique())

        for tp_idx, time_point in enumerate(time_points):
            phase = min(tp_idx // SAMPLES_PER_PHASE + 1, NUM_PHASES)

            # Get all squads at this time_point
            tp_df = match_df[match_df['time_point'] == time_point]

            preds = tp_df[pred_col].values
            targets = tp_df[target_col].values / TARGET_SCALE

            phase_data[phase].append({
                'preds': preds,
                'targets': targets,
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

            # Winner prediction
            pred_winner = np.argmax(preds)
            target_winner = np.argmax(targets)

            is_correct = pred_winner == target_winner
            if is_correct:
                phase_correct += 1
            phase_total += 1

            # Softmax for probabilities
            probs = softmax(preds)
            winner_prob = probs[target_winner]

            # Log loss
            winner_prob_clipped = np.clip(winner_prob, 1e-7, 1 - 1e-7)
            phase_log_losses.append(-np.log(winner_prob_clipped))

            # For ECE
            confidence = np.max(probs)
            phase_confidences.append(confidence)
            phase_accuracies.append(float(is_correct))

        # Phase accuracy
        if phase_total > 0:
            results['accuracy'][f'Phase_{phase}'] = phase_correct / phase_total
        else:
            results['accuracy'][f'Phase_{phase}'] = 0.0

        # Phase log loss
        if phase_log_losses:
            results['log_loss'][f'Phase_{phase}'] = np.mean(phase_log_losses)
        else:
            results['log_loss'][f'Phase_{phase}'] = 0.0

        # Phase ECE
        if phase_confidences:
            results['ece'][f'Phase_{phase}'] = compute_ece(
                np.array(phase_confidences),
                np.array(phase_accuracies),
            )
        else:
            results['ece'][f'Phase_{phase}'] = 0.0

        # Accumulate for average
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

    # Header
    header = f"{'Phase':<12}"
    for metric in metrics:
        header += f"{metric.upper():<15}"
    print(header)
    print("-" * 90)

    # Phase rows
    for phase in phases:
        row = f"{phase:<12}"
        for metric in metrics:
            val = results[metric].get(phase, 0.0)
            row += f"{val:<15.4f}"
        print(row)

    # Average row
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

    # Combined summary
    summary_data = []
    for metric in ['accuracy', 'log_loss', 'ece']:
        row = {'metric': metric}
        row.update(results[metric])
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="LightGBM Baseline for PGC")
    parser.add_argument("--folder_path", type=str, required=True,
                        help="Path to data folder")
    parser.add_argument("--split_csv_path", type=str, required=True,
                        help="Path to split CSV")
    parser.add_argument("--output_dir", type=str, default="results/lgbm",
                        help="Output directory")
    parser.add_argument("--toy_ratio", type=float, default=1.0,
                        help="Ratio of data to use (0.0-1.0)")
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_boost_round", type=int, default=1000)
    parser.add_argument("--use_dataset_v2", type=str, default="true",
                        choices=["true", "false"],
                        help="Use distance features (default: true)")
    args = parser.parse_args()

    # Convert string to boolean
    use_dataset_v2 = args.use_dataset_v2.lower() == "true"
    print(f"use_dataset_v2: {use_dataset_v2}")

    # Load data
    train_df, val_df, test_df, scaler_params, all_features = load_data(
        folder_path=args.folder_path,
        split_csv_path=args.split_csv_path,
        toy_ratio=args.toy_ratio,
        use_dataset_v2=use_dataset_v2,
    )

    # Train
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
    }

    model = train_lgbm(train_df, val_df, all_features, params)

    # Predict
    print("\nPredicting on test set...")
    test_df['pred'] = model.predict(test_df[all_features])

    # Evaluate
    print("\nEvaluating...")
    results = compute_metrics_by_phase(test_df)

    # Print and save results
    print_results(results, title="LightGBM Test Results")

    os.makedirs(args.output_dir, exist_ok=True)
    save_results(results, os.path.join(args.output_dir, "test_results.csv"))

    # Save model
    model_path = os.path.join(args.output_dir, "model.txt")
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")

    # Save config
    config = {
        'use_dataset_v2': use_dataset_v2,
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'num_boost_round': args.num_boost_round,
        'num_features': len(all_features),
        'features': all_features,
    }
    import json
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")


if __name__ == "__main__":
    main()


