# Combined Crawler Dataset — Closed-World

Run: 2026-03-09T16:20:54.040433

## Dataset
- Source: `/mnt/d/cs244c-gcp-only` (suffix: Combined)
- Classes: 95
- Samples: train=28592, valid=3574, test=3574
- Input length: 5000

## Hyperparameters
- Optimizer: Adamax (lr=0.002)
- Batch size: 128
- Max epochs: 50, ran 47
- EarlyStopping patience: 7
- ReduceLROnPlateau: factor=0.5, patience=3

## Results
- Best val accuracy: 0.4383 (epoch 43)
- Test accuracy: 0.4463
- Test loss: 2.2648
- Per-class accuracy: mean=0.3753, min=0.0000

## Model
- Architecture: DFNet (Sirinam et al. CCS'18)
- Best checkpoint: `/home/mswisher/cs244c/src/../saved_models/DF_Combined_best.keras`
