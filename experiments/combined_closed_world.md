# Combined Crawler Dataset — Closed-World

Run: 2026-03-13T18:10:04.026197

## Dataset
- Source: `/mnt/d/cs244c-cell-level` (suffix: Combined)
- Classes: 95
- Samples: train=72280, valid=9035, test=9035
- Input length: 5000

## Hyperparameters
- Optimizer: Adamax (lr=0.002)
- Batch size: 128
- Max epochs: 30, ran 30
- EarlyStopping patience: 7
- ReduceLROnPlateau: factor=0.5, patience=3

## Results
- Best val accuracy: 0.4609 (epoch 28)
- Test accuracy: 0.4804
- Test loss: 2.1849
- Per-class accuracy: mean=0.4708, min=0.0222

## Model
- Architecture: DFNet (Sirinam et al. CCS'18)
- Best checkpoint: `/home/mswisher/cs244c/src/../saved_models/DF_Combined_best.keras`
