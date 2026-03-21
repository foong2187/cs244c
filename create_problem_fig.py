#!/usr/bin/env python3
"""
WF Problem + Deep Fingerprinting figure — cream/cardinal theme.
Uses ax.text with bbox for properly sized boxes.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(42)
np.random.seed(42)

BG = '#F5F0E8'
CARDINAL = '#C41E3A'
CARDINAL_DIM = '#9A1830'
TEXT_DARK = '#3A2A2A'
TEXT_MID = '#7A6A60'
TEXT_FAINT = '#B0A498'
GRID_LINE = '#D8D0C4'
TEAL = '#2A7B9B'
GREEN_DARK = '#2D6B4A'
ORANGE_DARK = '#B85A1A'

plt.rcParams.update({
    'figure.facecolor': BG, 'text.color': TEXT_DARK,
    'font.family': 'Avenir Next', 'font.size': 11,
})

BBOX_STYLE = dict(boxstyle='round,pad=0.4', linewidth=1.5)


def create_figure():
    fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # ═══════════════════════════════════════
    # TOP: Attack Scenario
    # ═══════════════════════════════════════

    ax.text(8, 7.7, 'WEBSITE FINGERPRINTING ATTACK',
            ha='center', va='top', fontsize=20, fontweight='bold', color=CARDINAL)

    fy = 6.5  # flow y

    # User
    ax.text(1.5, fy, 'User', ha='center', va='center', fontsize=12, fontweight='bold',
            color=TEAL, bbox=dict(**BBOX_STYLE, facecolor='#EAF4F8', edgecolor=TEAL))

    # Arrow User → Tor
    ax.annotate('', xy=(3.5, fy), xytext=(2.2, fy),
                arrowprops=dict(arrowstyle='->', color=TEXT_MID, lw=1.3))
    ax.text(2.85, fy + 0.25, 'encrypted', ha='center', fontsize=7, color=GREEN_DARK, fontstyle='italic')

    # Tor Network
    ax.text(5.5, fy, 'Tor Network\n', ha='center', va='center', fontsize=13, fontweight='bold',
            color=GREEN_DARK, bbox=dict(boxstyle='round,pad=0.6', facecolor='#F0EDE5',
                                         edgecolor=GREEN_DARK, linewidth=1.5, linestyle='--'))
    ax.text(5.5, fy - 0.35, '3 encrypted hops', ha='center', fontsize=8, color=TEXT_MID)

    # Arrow Tor → Website
    ax.annotate('', xy=(8.3, fy), xytext=(7.2, fy),
                arrowprops=dict(arrowstyle='->', color=TEXT_MID, lw=1.3))
    ax.text(7.75, fy + 0.25, 'encrypted', ha='center', fontsize=7, color=GREEN_DARK, fontstyle='italic')

    # Website
    ax.text(9.2, fy, 'Website', ha='center', va='center', fontsize=12, fontweight='bold',
            color=GREEN_DARK, bbox=dict(**BBOX_STYLE, facecolor='#EAF8EE', edgecolor=GREEN_DARK))

    # ── Eavesdropper ──
    ey = 5.2
    ax.text(2.2, ey, 'Eavesdropper\n(local network)', ha='center', va='center',
            fontsize=10, fontweight='bold', color=CARDINAL,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FAEAEC',
                      edgecolor=CARDINAL, linewidth=1.5))

    # Tap arrow
    ax.annotate('', xy=(2.0, fy - 0.5), xytext=(2.0, ey + 0.5),
                arrowprops=dict(arrowstyle='->', color=CARDINAL, lw=1.2, linestyle='--'))
    ax.text(1.7, 5.85, 'taps', ha='right', fontsize=8, color=CARDINAL, fontstyle='italic')

    # Arrow to metadata
    ax.annotate('', xy=(5.0, ey), xytext=(3.3, ey),
                arrowprops=dict(arrowstyle='->', color=CARDINAL, lw=1))
    ax.text(4.1, ey + 0.2, 'sees', ha='center', fontsize=8, color=CARDINAL, fontstyle='italic')

    # Metadata
    ax.text(6.8, ey, ' timing  |  size  |  direction ', ha='center', va='center',
            fontsize=11, fontweight='bold', color=CARDINAL, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDF2F3',
                      edgecolor=CARDINAL, linewidth=1, linestyle=':'))
    ax.text(6.8, ey - 0.4, 'visible despite encryption', ha='center', fontsize=7.5, color=TEXT_MID)

    # ═══════════════════════════════════════
    # Divider
    # ═══════════════════════════════════════
    ax.plot([0.5, 15.5], [4.4, 4.4], '-', color=GRID_LINE, linewidth=0.8)

    # ═══════════════════════════════════════
    # BOTTOM: Deep Fingerprinting Pipeline
    # ═══════════════════════════════════════

    ax.text(8, 4.2, 'Deep Fingerprinting  (Sirinam et al., 2018)',
            ha='center', va='top', fontsize=16, fontweight='bold', color=CARDINAL)

    py = 2.8  # pipeline y

    # ── Direction sequence ──
    seq_x = 0.8
    seq_len = 50
    bw = 0.055
    dirs = [-1] * seq_len
    for i in [0,1,2,9,10,18,19,27,35,36,44]:
        if i < seq_len:
            dirs[i] = 1

    for i, d in enumerate(dirs):
        bx = seq_x + i * 0.08
        ax.bar(bx, 0.45 * d, bottom=py, width=bw,
               color=CARDINAL if d == 1 else TEAL, zorder=3)

    seq_end = seq_x + seq_len * 0.08
    ax.plot([seq_x - 0.1, seq_end + 0.1], [py, py], '-', color=TEXT_FAINT, linewidth=0.4)

    ax.text((seq_x + seq_end) / 2, py + 0.7, 'Packet Direction Sequence',
            ha='center', fontsize=10, fontweight='bold', color=TEXT_DARK)
    ax.text(seq_x - 0.15, py + 0.3, '+1', fontsize=7, color=CARDINAL, ha='right', fontweight='bold')
    ax.text(seq_x - 0.15, py - 0.3, '-1', fontsize=7, color=TEAL, ha='right', fontweight='bold')
    ax.text(seq_end + 0.2, py + 0.2, 'out', fontsize=6, color=CARDINAL)
    ax.text(seq_end + 0.2, py - 0.25, 'in', fontsize=6, color=TEAL)

    # Arrow → CNN
    arr_start = seq_end + 0.5
    ax.annotate('', xy=(arr_start + 0.5, py), xytext=(arr_start, py),
                arrowprops=dict(arrowstyle='->', color=TEXT_MID, lw=1.3))

    # ── CNN layers ──
    cnn_start = arr_start + 0.7
    widths =  [0.5, 0.45, 0.4, 0.35, 0.3]
    heights = [1.6, 1.3, 1.1, 0.8, 0.6]
    colors =  [CARDINAL, CARDINAL_DIM, CARDINAL_DIM, '#6A3040', '#4A2030']
    labels =  ['Conv', 'Conv', 'Conv', 'FC', 'FC']

    ax.text(cnn_start + 1.5, py + 1.05, '1D CNN', ha='center', fontsize=9,
            color=TEXT_MID)

    lx = cnn_start
    for j in range(5):
        from matplotlib.patches import FancyBboxPatch
        r = FancyBboxPatch((lx, py - heights[j]/2), widths[j], heights[j],
                            boxstyle="round,pad=0.02", facecolor=colors[j],
                            edgecolor='white', linewidth=0.4, alpha=0.85, zorder=3)
        ax.add_patch(r)
        rot = 90 if heights[j] > 0.9 else 0
        ax.text(lx + widths[j]/2, py, labels[j], ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold', zorder=4, rotation=rot)
        lx += widths[j] + 0.2

    # Arrow → softmax → prediction
    ax.annotate('', xy=(lx + 0.5, py), xytext=(lx + 0.05, py),
                arrowprops=dict(arrowstyle='->', color=TEXT_MID, lw=1.3))
    ax.text(lx + 0.27, py + 0.25, 'softmax', ha='center', fontsize=7,
            color=TEXT_FAINT, fontstyle='italic')

    # Prediction
    pred_x = lx + 1.5
    ax.text(pred_x, py + 0.15, 'youtube.com', ha='center', va='bottom',
            fontsize=13, fontweight='bold', color=CARDINAL, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FAEAEC',
                      edgecolor=CARDINAL, linewidth=1.8))
    ax.text(pred_x, py - 0.25, '98% accuracy', ha='center', va='top',
            fontsize=10, color=CARDINAL_DIM, fontweight='bold')

    # ── Bottom callouts ──
    ax.text(3, 0.7, 'Key Insight', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=ORANGE_DARK)
    ax.text(3, 0.7, '\nEncryption hides content but not traffic patterns',
            ha='center', va='top', fontsize=9, color=TEXT_DARK,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=BG,
                      edgecolor=ORANGE_DARK, linewidth=1.3))

    ax.text(13, 0.7, 'Open Question', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=TEXT_MID)
    ax.text(13, 0.7, '\nResult is from 2018 -- does it still hold today?',
            ha='center', va='top', fontsize=9, color=TEXT_MID,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=BG,
                      edgecolor=TEXT_FAINT, linewidth=1.2))

    fig.savefig('/Users/yousefh/.21st/worktrees/df-website-fingerprinting/yearning-harbor/problem-fig-plt.png',
                dpi=150, facecolor=BG, bbox_inches='tight', pad_inches=0.3)
    print('Saved problem-fig-plt.png')
    plt.close()


if __name__ == '__main__':
    create_figure()
