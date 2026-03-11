# Combined Crawler Dataset — Closed-World

Run: 2026-03-10T19:27:17.910934

## Dataset
- Source: `/mnt/d/cs244c-combined-maxacc` (suffix: Combined)
- Classes: 30
- Samples: train=6760, valid=845, test=845
- Input length: 5000

## Hyperparameters
- Optimizer: Adamax (lr=0.002)
- Batch size: 128
- Max epochs: 50, ran 50
- EarlyStopping patience: 7
- ReduceLROnPlateau: factor=0.5, patience=3

## Results
- Best val accuracy: 0.6283 (epoch 49)
- Test accuracy: 0.5988
- Test loss: 1.3807
- Per-class accuracy: mean=0.5999, min=0.3333

## Model
- Architecture: DFNet (Sirinam et al. CCS'18)
- Best checkpoint: `/home/mswisher/cs244c/src/../saved_models/DF_Combined_best.keras`
