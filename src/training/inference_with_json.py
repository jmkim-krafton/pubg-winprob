"""
Inference module with JSON output for PGC models.

This module runs inference on a single CSV file and saves detailed results
including win probabilities and squad state information for each time point.

Usage:
    from src.training.inference_with_json import run_inference_and_save_json
    
    results = run_inference_and_save_json(
        csv_path='path/to/match.csv',
        checkpoint_path='path/to/best.pt',
        output_dir='path/to/output/',
        device='cuda'
    )
"""

import os
import json
import ast
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data import CONTINUOUS_FEATURES, PGCDataset
from src.data.dataset_v2 import PGCDatasetV2, COMPUTED_FEATURES
from src.models import TransformerBackbone
from src.models.heads import get_head


# ============================================================================
# Zone Info Parsing
# ============================================================================

_ZONE_FLOAT_RE = re.compile(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?')


def parse_zone_info(zone_str: str) -> Tuple[float, float, float]:
    """
    Parse bluezone_info or whitezone_info string to tuple.
    
    Handles formats like:
    - "(408000.0, 408000.0, 581999.125)"
    - "(np.float64(x), np.float64(y), np.float64(z))"
    
    Returns:
        Tuple of (x, y, radius) as floats.
    """
    if pd.isna(zone_str) or zone_str == '':
        return 400000.0, 400000.0, 500000.0
    
    if isinstance(zone_str, (tuple, list)):
        if len(zone_str) >= 3:
            return float(zone_str[0]), float(zone_str[1]), float(zone_str[2])
        return 400000.0, 400000.0, 500000.0
    
    if isinstance(zone_str, str):
        # Remove np.float64(), np.float32(), etc.
        cleaned = re.sub(r'np\.float\d+\s*\(\s*([^)]+)\s*\)', r'\1', zone_str)
        nums = _ZONE_FLOAT_RE.findall(cleaned)
        if len(nums) >= 3:
            return float(nums[0]), float(nums[1]), float(nums[2])
    
    return 400000.0, 400000.0, 500000.0


def parse_positions(positions_str: str) -> List[List[float]]:
    """
    Parse positions string to list of coordinates.
    
    Args:
        positions_str: String like "[[x1,y1,z1], [x2,y2,z2], ...]"
    
    Returns:
        List of [x, y, z] coordinates for each player.
    """
    if pd.isna(positions_str) or positions_str == '':
        return [[np.nan, np.nan, np.nan]]
    
    # Replace 'nan' with 'None' for ast.literal_eval
    positions_str_clean = positions_str.replace('nan', 'None')
    positions_list = ast.literal_eval(positions_str_clean)
    
    result = []
    for pos in positions_list:
        if pos is None or any(p is None for p in pos):
            result.append([np.nan, np.nan, np.nan])
        elif pos[0] == 0 and pos[1] == 0 and pos[2] == 0:
            result.append([np.nan, np.nan, np.nan])
        else:
            result.append([float(pos[0]), float(pos[1]), float(pos[2])])
    
    return result if result else [[np.nan, np.nan, np.nan]]


# ============================================================================
# Model Loading
# ============================================================================

def load_temperature(checkpoint_path: str) -> Optional[float]:
    """Load temperature parameter from checkpoint directory if available."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    tau_path = os.path.join(checkpoint_dir, "temperature.json")
    
    if os.path.exists(tau_path):
        with open(tau_path, "r") as f:
            tau_data = json.load(f)
        return tau_data.get("optimal_tau", None)
    
    return None


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str = None,
    device: str = "cpu",
) -> Tuple[nn.Module, nn.Module, Dict, Dict]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., best.pt).
        config_path: Path to config.json. If None, inferred from checkpoint_path.
        device: Device to load model on.
    
    Returns:
        Tuple of (backbone, head, scaler_params, config).
    """
    # Load config
    if config_path is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.json")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Determine input dimension
    use_dataset_v2 = config.get("use_dataset_v2", True)
    if use_dataset_v2:
        input_dim = config.get("input_dim", len(CONTINUOUS_FEATURES) + len(COMPUTED_FEATURES))
    else:
        input_dim = config.get("input_dim", len(CONTINUOUS_FEATURES))
    
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    
    backbone = TransformerBackbone(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    
    # Get appropriate head based on loss type from config
    loss_type = config.get("loss_type", "mse")
    head = get_head(loss_type, embed_dim=embed_dim, dropout=dropout)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    head.load_state_dict(checkpoint["head_state_dict"])
    scaler_params = checkpoint.get("scaler_params", None)
    
    backbone = backbone.to(device)
    head = head.to(device)
    
    return backbone, head, scaler_params, config


# ============================================================================
# Inference Functions
# ============================================================================

def run_inference_on_csv(
    csv_path: str,
    backbone: nn.Module,
    head: nn.Module,
    scaler_params: Dict,
    config: Dict,
    device: str = "cpu",
    temperature: float = 1.0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Run inference on a single CSV file.
    
    Args:
        csv_path: Path to input CSV file.
        backbone: Loaded backbone model.
        head: Loaded head model.
        scaler_params: Scaler parameters for feature normalization.
        config: Model configuration.
        device: Device to use.
        temperature: Temperature for probability calibration.
    
    Returns:
        Tuple of (all_predicted_probabilities dict, raw DataFrame).
    """
    # Determine which dataset class to use
    use_dataset_v2 = config.get("use_dataset_v2", True)
    
    # Handle case where csv_path is in current directory
    csv_path = os.path.abspath(csv_path)
    folder_path = os.path.dirname(csv_path)
    file_name = os.path.basename(csv_path)
    
    if use_dataset_v2:
        dataset = PGCDatasetV2(
            folder_path=folder_path,
            file_list=[file_name],
            continuous_features=CONTINUOUS_FEATURES,
            scaler_params=scaler_params,
        )
    else:
        dataset = PGCDataset(
            folder_path=folder_path,
            file_list=[file_name],
            continuous_features=CONTINUOUS_FEATURES,
            scaler_params=scaler_params,
        )
    
    # Load raw data (unscaled) for additional info
    raw_df = pd.read_csv(csv_path)
    
    # Build a lookup dict: (match_id, time_point_rounded) -> DataFrame rows
    # Round time_point to avoid float comparison issues
    raw_df['time_point_key'] = raw_df['time_point'].round(6)
    time_point_groups = {}
    for (mid, tp_key), group in raw_df.groupby(['match_id', 'time_point_key']):
        time_point_groups[(mid, tp_key)] = group.sort_values('squad_number')
    
    backbone.eval()
    head.eval()
    
    all_predicted_probabilities = {}
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            match_id, time_point = dataset.group_keys[idx]
            sample = dataset[idx]
            
            # Prepare batch (add batch dimension)
            x = sample["x_continuous"].unsqueeze(0).to(device)
            positions = sample["positions"].unsqueeze(0).to(device)
            bluezone = sample["bluezone_info"].unsqueeze(0).to(device)
            whitezone = sample["whitezone_info"].unsqueeze(0).to(device)
            map_idx = sample["map_idx"].unsqueeze(0).to(device)
            num_squads = sample["num_squads"]
            
            # Get raw data for this time point using rounded key
            time_point_key = round(time_point, 6)
            time_data = time_point_groups.get((match_id, time_point_key))
            
            if time_data is None:
                print(f"Warning: No raw data found for {match_id}, {time_point}")
                continue
            
            # Use only num_squads rows (matches dataset)
            time_data = time_data.head(num_squads)
            squad_numbers = time_data['squad_number'].tolist()
            
            # Get raw (unscaled) values
            if 'squad_alive_count' in time_data.columns:
                alive_cnt = [int(x) for x in time_data['squad_alive_count'].tolist()]
            else:
                alive_cnt = [None] * len(squad_numbers)
            
            if 'squad_total_health' in time_data.columns:
                hp = [int(x) for x in time_data['squad_total_health'].tolist()]
            else:
                hp = [None] * len(squad_numbers)
            
            # Parse positions first (needed for alive check)
            positions_list = []
            if 'positions' in time_data.columns:
                for pos_str in time_data['positions'].tolist():
                    positions_list.append(parse_positions(pos_str))
            
            # Build squad_alive_mask from raw data (more reliable than dataset mask)
            # A squad is dead if:
            #   - squad_alive_count == 0, OR
            #   - squad_total_health == 0, OR
            #   - all positions are NaN
            squad_alive_mask = torch.zeros(num_squads, dtype=torch.bool)
            for i in range(len(squad_numbers)):
                alive_count = alive_cnt[i] if alive_cnt[i] is not None else 0
                health = hp[i] if hp[i] is not None else 0
                
                # Check if all positions are NaN (dead squad)
                all_positions_nan = True
                if i < len(positions_list):
                    for pos in positions_list[i]:
                        if not (np.isnan(pos[0]) and np.isnan(pos[1]) and np.isnan(pos[2])):
                            all_positions_nan = False
                            break
                
                squad_alive_mask[i] = (alive_count > 0) and (health > 0) and (not all_positions_nan)
            
            # Forward pass
            embeddings = backbone(x, positions, bluezone, whitezone, map_idx)
            raw_scores = head(embeddings).squeeze(0).cpu()
            
            # Apply temperature scaling
            scaled_scores = raw_scores / temperature
            
            # Compute win probabilities (softmax over alive squads)
            masked_scores = scaled_scores.clone()
            masked_scores[~squad_alive_mask] = float("-inf")
            win_probs = torch.softmax(masked_scores[:num_squads], dim=0)
            
            # Parse zone info
            if 'bluezone_info' in time_data.columns and not time_data.empty:
                bluezone_str = time_data['bluezone_info'].iloc[0]
                bluezone_parsed = parse_zone_info(bluezone_str)
            else:
                bluezone_parsed = None
            
            if 'whitezone_info' in time_data.columns and not time_data.empty:
                whitezone_str = time_data['whitezone_info'].iloc[0]
                whitezone_parsed = parse_zone_info(whitezone_str)
            else:
                whitezone_parsed = None
            
            # Get phase info
            if 'phase' in time_data.columns and not time_data.empty:
                phase = int(time_data['phase'].iloc[0])
            else:
                phase = None
            
            # Build probabilities dict (dead squads get 0.0 probability)
            probabilities = {}
            for i, squad_num in enumerate(squad_numbers):
                if i < len(win_probs):
                    # Explicitly set dead squad probability to 0.0
                    if squad_alive_mask[i]:
                        probabilities[int(squad_num)] = float(win_probs[i].item())
                    else:
                        probabilities[int(squad_num)] = 0.0
            
            # Build is_alive dict (consistent with squad_alive_mask)
            is_alive = {}
            for i, squad_num in enumerate(squad_numbers):
                is_alive[int(squad_num)] = bool(squad_alive_mask[i])
            
            # Initialize match_id dict if needed
            if match_id not in all_predicted_probabilities:
                all_predicted_probabilities[match_id] = {}
            
            # Store time point data
            all_predicted_probabilities[match_id][str(time_point)] = {
                'phase': phase,
                'squad_numbers': [int(sn) for sn in squad_numbers],
                'alive_cnt': alive_cnt,
                'hp': hp,
                'bluezone_info': {
                    'x': bluezone_parsed[0] if bluezone_parsed else None,
                    'y': bluezone_parsed[1] if bluezone_parsed else None,
                    'radius': bluezone_parsed[2] if bluezone_parsed else None,
                },
                'whitezone_info': {
                    'x': whitezone_parsed[0] if whitezone_parsed else None,
                    'y': whitezone_parsed[1] if whitezone_parsed else None,
                    'radius': whitezone_parsed[2] if whitezone_parsed else None,
                },
                'positions': positions_list,
                'probabilities': probabilities,
                'is_alive': is_alive,
            }
    
    # Clean up temporary column
    raw_df.drop('time_point_key', axis=1, inplace=True)
    
    return all_predicted_probabilities, raw_df


def determine_winner(df: pd.DataFrame) -> Optional[int]:
    """Determine the winner squad number from DataFrame."""
    # Priority 1: squad_win column
    if 'squad_win' in df.columns:
        winner_rows = df[df['squad_win'] == 1]
        if not winner_rows.empty:
            return int(winner_rows['squad_number'].iloc[0])
    
    # Priority 2: squad_ranking column
    if 'squad_ranking' in df.columns:
        winner_rows = df[df['squad_ranking'] == 1]
        if not winner_rows.empty:
            return int(winner_rows['squad_number'].iloc[0])
    
    return None


def run_inference_and_save_json(
    csv_path: str,
    checkpoint_path: str,
    output_dir: str = None,
    config_path: str = None,
    device: str = "cpu",
    temperature: float = None,
) -> Dict[str, Any]:
    """
    Run inference on a single CSV file and save results to JSON.
    
    Args:
        csv_path: Path to input CSV file (single match).
        checkpoint_path: Path to model checkpoint (best.pt).
        output_dir: Directory to save output files. If None, uses checkpoint directory.
        config_path: Path to config.json. If None, inferred from checkpoint_path.
        device: Device to use ('cpu', 'cuda', 'cuda:0', etc.).
        temperature: Temperature for probability calibration. 
                     If None, loads from temperature.json or uses 1.0.
    
    Returns:
        Dictionary containing:
            - output_dir: Path to output directory
            - json_path: Path to saved JSON file
            - match_id: Match identifier
            - winner_squad: Winner squad number
            - num_time_points: Number of time points processed
            - all_predicted_probabilities: The full predictions dict
    """
    print("=" * 80)
    print("INFERENCE WITH JSON OUTPUT")
    print("=" * 80)
    print(f"CSV file: {csv_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    backbone, head, scaler_params, config = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )
    
    if scaler_params is None:
        raise ValueError("Checkpoint does not contain scaler_params.")
    
    # Load temperature
    if temperature is None:
        temperature = load_temperature(checkpoint_path)
        if temperature is not None:
            print(f"Using temperature from checkpoint: {temperature:.4f}")
        else:
            temperature = 1.0
            print("Using default temperature: 1.0")
    
    print(f"Model config: embed_dim={config['embed_dim']}, "
          f"num_heads={config['num_heads']}, num_layers={config['num_layers']}")
    print(f"Loss type: {config.get('loss_type', 'mse')}")
    
    # Run inference
    print("\nRunning inference...")
    all_predicted_probabilities, raw_df = run_inference_on_csv(
        csv_path=csv_path,
        backbone=backbone,
        head=head,
        scaler_params=scaler_params,
        config=config,
        device=device,
        temperature=temperature,
    )
    
    # Get match info
    match_ids = list(all_predicted_probabilities.keys())
    if len(match_ids) != 1:
        print(f"Warning: CSV contains {len(match_ids)} matches. Using first one.")
    
    match_id = match_ids[0]
    num_time_points = len(all_predicted_probabilities[match_id])
    winner_squad = determine_winner(raw_df)
    
    print(f"\nMatch ID: {match_id}")
    print(f"Winner squad: {winner_squad}")
    print(f"Time points processed: {num_time_points}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"predictions_{csv_basename}_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    # Build output data
    output_data = {
        'metadata': {
            'csv_file': os.path.basename(csv_path),
            'checkpoint': os.path.basename(checkpoint_path),
            'model_config': {
                'embed_dim': config['embed_dim'],
                'num_heads': config['num_heads'],
                'num_layers': config['num_layers'],
                'loss_type': config.get('loss_type', 'mse'),
            },
            'temperature': temperature,
            'device': device,
            'created_at': datetime.now().isoformat(),
        },
        'match_info': {
            'match_id': match_id,
            'winner_squad': winner_squad,
            'num_time_points': num_time_points,
            'map': raw_df['map'].iloc[0] if 'map' in raw_df.columns else None,
        },
        'predictions': all_predicted_probabilities,
    }
    
    # Save to JSON
    print(f"\nSaving results to: {json_path}")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    
    return {
        'output_dir': output_dir,
        'json_path': json_path,
        'match_id': match_id,
        'winner_squad': winner_squad,
        'num_time_points': num_time_points,
        'all_predicted_probabilities': all_predicted_probabilities,
    }


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run inference on a CSV file and save results to JSON."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to input CSV file (single match)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., checkpoints/exp/best.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output files. Default: checkpoint directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for probability calibration. Default: load from checkpoint or 1.0",
    )
    args = parser.parse_args()
    
    # Determine device
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda:0"
    elif args.device.startswith("cuda") and torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"
    
    results = run_inference_and_save_json(
        csv_path=args.csv_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        device=device,
        temperature=args.temperature,
    )
    
    print(f"\nResults saved to: {results['json_path']}")


