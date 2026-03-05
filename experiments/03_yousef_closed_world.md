# Yousef Closed-World Baseline

Run: 2026-03-04T13:37:16.446202

## Dataset
- Source: Yousef pickles (data/yousef-data/pickle or --data_dir)
- Classes: 100
- Input length: 5000

## Hyperparameters
- Epochs: 30, batch_size: 128, lr: 0.002
- Optimizer: Adamax (paper default)

## Results
- Validation accuracy (final): 0.1140
- Test accuracy: 0.1017
- Test loss: 3.7946

## Comparison
- GTT23 baseline (01): ~53% val accuracy (exit-relay data).
- Fresh2026 (02): ~16% test accuracy (92 classes).
- CCS'18 paper: ~98% on client-side entry-guard data.
