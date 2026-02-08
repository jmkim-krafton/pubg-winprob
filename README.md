# PUBG Win Probability Prediction

Real-time win probability prediction for PUBG esports matches using Transformer-based survival analysis. Given in-game features (combat stats, positions, zone dynamics) at any point during a match, the model predicts each squad's probability of winning (WWCD - Winner Winner Chicken Dinner).

## Project Structure

```
pubg-winprob/
├── src/
│   ├── data/                  # Data loading & feature definitions
│   │   ├── continuous_features.py   # 53 continuous feature definitions
│   │   ├── dataset.py               # PyTorch Dataset (v1)
│   │   ├── dataset_v2.py            # Enhanced dataset with zone distance features
│   │   └── utils.py                 # Position/zone parsing utilities
│   ├── models/                # Neural network architecture
│   │   ├── backbone.py              # Transformer backbone with positional encoding
│   │   ├── modules.py               # Fourier encoding, zone/token embeddings
│   │   └── heads.py                 # Prediction heads (survival, hazard, classification)
│   └── training/              # Training & evaluation pipeline
│       ├── trainer.py               # Trainer with DDP multi-GPU support
│       ├── losses.py                # Loss functions (MSE, Cox, Ranking, CE, etc.)
│       ├── metrics.py               # Winner accuracy, log loss, ECE
│       ├── evaluation.py            # Phase-wise test evaluation
│       ├── inference.py             # Checkpoint loading & batch inference
│       └── calibration.py           # Temperature scaling for probability calibration
├── scripts/
│   ├── train_mse_baseline.py        # Main training script
│   ├── lgbm_baseline.py             # LightGBM baseline
│   ├── run_experiments_v21.sh       # Hyperparameter grid search
│   └── run_inference.sh             # Inference entry point
├── data_generation/
│   ├── feature_engineering/         # Feature extraction from match logs
│   ├── generate_postmatch_dataset/  # Post-match JSON → CSV pipeline
│   └── generate_competitive_dataset/# Competitive match dataset generation
├── notebooks/
│   ├── run_files/                   # Training pipeline notebooks
│   ├── analysis/                    # Model evaluation & comparison
│   ├── baselines/                   # Rule-based & LightGBM baselines
│   ├── calibration/                 # Temperature scaling notebooks
│   └── inference/                   # Inference demos
├── log_data/
│   ├── dictionaries/               # Game metadata (weapons, items, maps)
│   ├── samples/                     # Example match JSON files
│   └── tournamentid/               # Tournament match ID lists
└── requirements.txt
```

## Model Architecture

- **Transformer backbone** with multi-head self-attention over squads (up to 16 per match)
- **Fourier positional encoding** for 3D player coordinates
- **Zone embedding** for bluezone/whitezone position and radius
- **Map embedding** for 4 PUBG maps (Erangel, Miramar, Taego, Rondo)
- **Multiple prediction heads**: survival time regression, Cox hazard, winner classification

## Supported Loss Functions

| Loss Type | Description |
|---|---|
| `mse` | Survival time regression |
| `cox` | Cox partial likelihood (full retrospective) |
| `rank_cox` | Cox + ranking consistency penalty |
| `weighted_cox` | Cox + time-gap weighting |
| `ce` | Cross-entropy classification |
| `concordance` | C-index with time-gap weighting |
| `survival_ce` | Survival score cross-entropy |

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch >= 2.0.0
- pandas, numpy, tqdm
- captum (for GradientSHAP analysis)
- matplotlib, seaborn (visualization)

## Usage

### Training

```bash
python scripts/train_mse_baseline.py \
    --train_folder /path/to/train \
    --test_folder /path/to/test \
    --loss_type weighted_cox \
    --embed_dim 256 --num_heads 4 --num_layers 2 \
    --batch_size 64 --epochs 50 --lr 1e-3 \
    --device cuda
```

Use `--toy_ratio 0.1` for quick testing with a subset of data.

### Hyperparameter Search

```bash
bash scripts/run_experiments_v21.sh
```

Runs a grid search over embedding dimensions, attention heads, layers, dropout, and learning rates.

### Inference

```bash
# Single CSV
bash scripts/run_inference.sh \
    --checkpoint_dir ./checkpoints/exp_001 \
    --csv_path ./test_data.csv

# Batch inference on folder
bash scripts/run_inference.sh \
    --checkpoint_dir ./checkpoints/exp_001 \
    --folder_path ./test_folder \
    --file_list "match1.csv match2.csv"
```

### LightGBM Baseline

```bash
python scripts/lgbm_baseline.py \
    --train_folder /path/to/train \
    --test_folder /path/to/test
```

## Data Format

Each match is represented as a CSV with **50 time points** (10 phases, 5 per phase). Features include:

- **Action features**: kills, damage dealt (by weapon type), healing, pickups
- **Spatial features**: player positions, distance to bluezone/whitezone
- **Squad status**: alive count, total health, armor/backpack levels
- **Zone dynamics**: bluezone/whitezone center and radius

Squads are padded to a maximum of 16 per match.

## Evaluation Metrics

- **Winner Accuracy**: whether the predicted winner matches the actual winner
- **Log Loss**: cross-entropy of predicted win probabilities
- **ECE (Expected Calibration Error)**: reliability of probability estimates
- **Phase-wise Accuracy**: prediction performance tracked across 10 match phases

## License

Apache License 2.0
