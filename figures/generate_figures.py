"""
Generate publication-quality figures for the DF website fingerprinting paper.

Figures:
  1. Defense Comparison Bar Chart (closed-world accuracy)
  3. Open-World Precision-Recall operating points (per defense, at tau=0.50)
  9. Training Curves (loss & accuracy over epochs, per defense)

All values from the paper's actual experimental results (Tables 1-2).

Usage:
    python figures/generate_figures.py            # Generate all figures
    python figures/generate_figures.py --fig 1    # Generate only Fig 1
    python figures/generate_figures.py --fig 3 9  # Generate Figs 3 and 9
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe

# ============================================================================
# Modern Style Configuration
# ============================================================================
BG = '#FAFAFA'
GRID_COLOR = '#E0E0E0'
TEXT_COLOR = '#2D2D2D'
SUBTLE = '#888888'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 600,
    'axes.labelcolor': TEXT_COLOR,
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 0.6,
    'axes.facecolor': BG,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.color': SUBTLE,
    'ytick.color': SUBTLE,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize': 9.5,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#DDDDDD',
    'legend.fancybox': True,
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linewidth': 0.4,
    'grid.color': GRID_COLOR,
    'grid.linestyle': '-',
    'text.color': TEXT_COLOR,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTDIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTDIR, exist_ok=True)

# Refined color palette — muted, modern tones
COLORS = {
    'NoDef':         '#3B82F6',  # bright blue
    'WalkieTalkie':  '#A855F7',  # purple
    'RegulaTor':     '#10B981',  # emerald
    'BRO':           '#F59E0B',  # amber
    'Tamaraw':       '#EF4444',  # red
    'BuFLO':         '#F97316',  # orange
}


def _clean_ax(ax):
    """Apply clean modern styling to an axes."""
    ax.tick_params(axis='both', which='both', length=3, width=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


# ============================================================================
# Fig 1 — Defense Comparison Bar Chart (Closed-World Accuracy)
# ============================================================================
def fig1_defense_comparison():
    """Horizontal bar chart of closed-world top-1 test accuracy per defense.

    Source: Paper Table 1.
    """
    defenses =   ['Tamaraw', 'BuFLO', 'WalkieTalkie', 'BRO', 'RegulaTor', 'NoDef']
    accuracies = [26.1,       31.7,    46.1,           74.1,  81.8,        96.1]

    colors = [COLORS[d] for d in defenses]

    fig, ax = plt.subplots(figsize=(7, 3.8))

    y = np.arange(len(defenses))
    bars = ax.barh(y, accuracies, height=0.55, color=colors,
                   edgecolor='none', zorder=3, alpha=0.9)

    # Rounded bar caps via path effects
    for bar in bars:
        bar.set_linewidth(0)

    # Value labels
    for bar, acc in zip(bars, accuracies):
        x_pos = bar.get_width() + 1.0
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{acc:.1f}%', ha='left', va='center', fontsize=10,
                fontweight='600', color=TEXT_COLOR)

    # Original paper reference line
    ax.axvline(x=98.3, color='#3B82F6', linestyle='--', linewidth=0.9,
               alpha=0.4, zorder=2)
    ax.text(98.3, len(defenses) - 0.3, 'Original\n98.3%', ha='center',
            va='bottom', fontsize=7.5, color='#3B82F6', alpha=0.7,
            fontweight='500')

    # Random guess
    ax.axvline(x=100.0 / 95, color=SUBTLE, linestyle=':', linewidth=0.7,
               alpha=0.5, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(defenses, fontsize=10.5, fontweight='500')
    ax.set_xlabel('Top-1 Test Accuracy (%)', fontweight='500')
    ax.set_xlim(0, 108)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.invert_yaxis()
    _clean_ax(ax)

    # Subtle title
    ax.set_title('Closed-World Classification Accuracy', fontweight='600',
                 pad=12)

    fig.tight_layout()
    path = os.path.join(OUTDIR, 'fig1_defense_comparison.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    print(f'Fig 1 saved to {path}')
    plt.close(fig)


# ============================================================================
# Fig 3 — Open-World Precision vs Recall (operating points at tau=0.50)
# ============================================================================
def fig3_precision_recall():
    """Open-world precision vs recall scatter with annotations.

    Source: Paper Table 2.
    """
    data = {
        'NoDef':      (0.9895, 0.9989, 0.9895, 0.0010),
        'RegulaTor':  (0.8189, 0.9898, 0.8189, 0.0080),
        'BRO':        (0.7284, 0.9105, 0.7284, 0.0680),
        'BuFLO':      (0.1432, 0.7473, 0.1432, 0.0460),
        'Tamaraw':    (0.0747, 0.7396, 0.0747, 0.0250),
    }

    fig, ax = plt.subplots(figsize=(6, 5))

    # Connect points first (behind)
    recalls =    [d[0] for d in data.values()]
    precisions = [d[1] for d in data.values()]
    order = np.argsort(recalls)
    ax.plot(np.array(recalls)[order], np.array(precisions)[order],
            color='#D1D5DB', linestyle='-', linewidth=1.5, alpha=0.8, zorder=1)

    for defense, (recall, precision, tpr, fpr) in data.items():
        ax.scatter(recall, precision,
                   color=COLORS[defense], s=140, zorder=4,
                   edgecolors='white', linewidths=2)
        # Shadow ring
        ax.scatter(recall, precision,
                   color='none', s=160, zorder=3,
                   edgecolors=COLORS[defense], linewidths=0.5, alpha=0.3)

        offsets = {
            'NoDef':     (0.00,  -0.030),
            'RegulaTor': (-0.12, -0.020),
            'BRO':       (0.06,  -0.030),
            'BuFLO':     (0.12,   0.008),
            'Tamaraw':   (-0.06,  0.020),
        }
        dx, dy = offsets[defense]
        ax.annotate(
            f'{defense}',
            (recall, precision),
            xytext=(recall + dx, precision + dy),
            fontsize=8.5, ha='center', fontweight='500', color=TEXT_COLOR,
            arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.5,
                            connectionstyle='arc3,rad=0.15'),
        )
        # FPR as smaller subtitle
        ax.text(recall + dx, precision + dy - 0.018,
                f'FPR={fpr:.3f}', fontsize=6.5, ha='center', color=SUBTLE)

    ax.set_xlabel('Recall (TPR)', fontweight='500')
    ax.set_ylabel('Precision', fontweight='500')
    ax.set_title(r'Open-World Performance at $\tau = 0.50$', fontweight='600',
                 pad=12)
    ax.set_xlim(-0.04, 1.10)
    ax.set_ylim(0.68, 1.025)
    _clean_ax(ax)

    # Legend with colored dots
    for defense in data:
        ax.plot([], [], 'o', color=COLORS[defense], markersize=7,
                label=defense, markeredgecolor='white', markeredgewidth=1)
    ax.legend(loc='lower left', frameon=True, borderpad=0.8,
              handletextpad=0.5, labelspacing=0.6)

    fig.tight_layout()
    path = os.path.join(OUTDIR, 'fig3_open_world_pr.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    print(f'Fig 3 saved to {path}')
    plt.close(fig)


# ============================================================================
# Fig 9 — Training Curves (Loss & Accuracy over Epochs)
# ============================================================================
def fig9_training_curves():
    """Training loss and validation accuracy over epochs.

    Data extracted from actual training run on self-collected NoDef dataset.
    Title: "DFNet convergence (max-accuracy subset)"
    Best val accuracy: 0.628 at epoch 49
    Best val loss: 1.380 at epoch 49
    50 epochs total.
    """
    epochs = np.arange(1, 51)

    # ---- REAL DATA: extracted from training plots ----
    # Accuracy (from "DFNet convergence (max-accuracy subset)")
    train_acc = np.array([
        0.070, 0.088, 0.100, 0.120, 0.148, 0.178, 0.215, 0.250, 0.280, 0.305,
        0.320, 0.335, 0.350, 0.370, 0.395, 0.415, 0.430, 0.445, 0.460, 0.480,
        0.500, 0.518, 0.535, 0.550, 0.565, 0.580, 0.593, 0.605, 0.618, 0.633,
        0.645, 0.655, 0.663, 0.672, 0.682, 0.690, 0.700, 0.708, 0.715, 0.722,
        0.728, 0.733, 0.738, 0.742, 0.745, 0.748, 0.750, 0.752, 0.753, 0.755,
    ])
    val_acc = np.array([
        0.030, 0.038, 0.048, 0.058, 0.080, 0.103, 0.150, 0.248, 0.255, 0.300,
        0.370, 0.382, 0.370, 0.378, 0.420, 0.432, 0.435, 0.440, 0.443, 0.465,
        0.530, 0.540, 0.500, 0.520, 0.558, 0.565, 0.555, 0.550, 0.560, 0.570,
        0.563, 0.565, 0.580, 0.590, 0.598, 0.583, 0.590, 0.600, 0.593, 0.600,
        0.603, 0.608, 0.615, 0.618, 0.620, 0.623, 0.625, 0.626, 0.628, 0.625,
    ])

    # Loss (from "DFNet loss (max-accuracy subset)")
    train_loss = np.array([
        3.770, 3.380, 3.100, 2.950, 2.820, 2.720, 2.630, 2.540, 2.460, 2.380,
        2.310, 2.250, 2.190, 2.130, 2.070, 2.010, 1.960, 1.900, 1.850, 1.790,
        1.730, 1.680, 1.630, 1.580, 1.540, 1.500, 1.460, 1.420, 1.380, 1.340,
        1.300, 1.270, 1.240, 1.210, 1.180, 1.150, 1.120, 1.095, 1.070, 1.045,
        1.020, 1.000, 0.980, 0.960, 0.945, 0.930, 0.915, 0.900, 0.888, 0.875,
    ])
    val_loss = np.array([
        4.000, 4.220, 4.050, 3.800, 3.580, 3.300, 2.900, 2.730, 2.720, 2.250,
        2.220, 2.210, 2.250, 2.150, 2.000, 1.990, 1.975, 1.950, 1.900, 1.700,
        1.680, 1.730, 1.650, 1.600, 1.560, 1.545, 1.520, 1.500, 1.510, 1.490,
        1.520, 1.540, 1.500, 1.480, 1.460, 1.475, 1.460, 1.440, 1.430, 1.425,
        1.415, 1.410, 1.400, 1.395, 1.390, 1.388, 1.385, 1.383, 1.380, 1.382,
    ])

    BLUE = '#3B82F6'
    ORANGE = '#F59E0B'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- Left: Loss ----
    ax1.plot(epochs, train_loss, color=BLUE, linewidth=1.8, label='Train',
             path_effects=[pe.Stroke(linewidth=3.5, foreground='white', alpha=0.5),
                           pe.Normal()])
    ax1.plot(epochs, val_loss, color=ORANGE, linewidth=1.8, label='Validation',
             path_effects=[pe.Stroke(linewidth=3.5, foreground='white', alpha=0.5),
                           pe.Normal()])

    # Mark best val loss
    best_loss_ep = 49
    best_loss_val = 1.380
    ax1.scatter([best_loss_ep], [best_loss_val], color=ORANGE, s=50, zorder=5,
                edgecolors='white', linewidths=1.5)
    ax1.annotate(f'Best: {best_loss_val:.3f}\n(epoch {best_loss_ep})',
                 (best_loss_ep, best_loss_val),
                 xytext=(best_loss_ep - 14, best_loss_val + 0.30),
                 fontsize=8, color=SUBTLE, fontweight='500',
                 arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=0.6))

    ax1.set_xlabel('Epoch', fontweight='500')
    ax1.set_ylabel('Cross-Entropy Loss', fontweight='500')
    ax1.set_title('Training Loss', fontweight='600', pad=10)
    ax1.set_xlim(0, 51)
    ax1.set_ylim(0, 4.5)
    ax1.legend(loc='upper right', frameon=True, borderpad=0.6)
    _clean_ax(ax1)

    # ---- Right: Accuracy ----
    ax2.plot(epochs, train_acc * 100, color=BLUE, linewidth=1.8, label='Train',
             path_effects=[pe.Stroke(linewidth=3.5, foreground='white', alpha=0.5),
                           pe.Normal()])
    ax2.plot(epochs, val_acc * 100, color=ORANGE, linewidth=1.8, label='Validation',
             path_effects=[pe.Stroke(linewidth=3.5, foreground='white', alpha=0.5),
                           pe.Normal()])

    # Mark best val accuracy
    best_acc_ep = 49
    best_acc_val = 0.628
    ax2.scatter([best_acc_ep], [best_acc_val * 100], color=ORANGE, s=50, zorder=5,
                edgecolors='white', linewidths=1.5)
    ax2.annotate(f'Best: {best_acc_val:.1%}\n(epoch {best_acc_ep})',
                 (best_acc_ep, best_acc_val * 100),
                 xytext=(best_acc_ep - 14, best_acc_val * 100 + 8),
                 fontsize=8, color=SUBTLE, fontweight='500',
                 arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=0.6))

    # Overfitting gap shading
    ax2.fill_between(epochs, val_acc * 100, train_acc * 100,
                     alpha=0.07, color=ORANGE, zorder=1)

    ax2.set_xlabel('Epoch', fontweight='500')
    ax2.set_ylabel('Accuracy (%)', fontweight='500')
    ax2.set_title('Classification Accuracy', fontweight='600', pad=10)
    ax2.set_xlim(0, 51)
    ax2.set_ylim(0, 82)
    ax2.yaxis.set_major_locator(MultipleLocator(10))
    ax2.legend(loc='lower right', frameon=True, borderpad=0.6)
    _clean_ax(ax2)

    fig.suptitle('DFNet Convergence (Self-Collected NoDef)', fontweight='600',
                 fontsize=14, y=1.02)
    fig.tight_layout(w_pad=3)

    path = os.path.join(OUTDIR, 'fig9_training_curves.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    print(f'Fig 9 saved to {path}')
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
FIGURE_MAP = {
    '1': ('Defense Comparison Bar Chart', fig1_defense_comparison),
    '3': ('Open-World Precision-Recall', fig3_precision_recall),
    '9': ('Training Curves', fig9_training_curves),
}


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--fig', nargs='*', default=None,
                        help='Figure numbers to generate (default: all)')
    args = parser.parse_args()

    figs = args.fig if args.fig else list(FIGURE_MAP.keys())

    for fig_num in figs:
        if fig_num not in FIGURE_MAP:
            print(f'Unknown figure: {fig_num}. Options: {list(FIGURE_MAP.keys())}')
            continue
        name, fn = FIGURE_MAP[fig_num]
        print(f'\n--- Generating Fig {fig_num}: {name} ---')
        fn()

    print(f'\nAll figures saved to {OUTDIR}/')


if __name__ == '__main__':
    main()
