#!/usr/bin/env python3
"""
WF Defense Visualizations — Matplotlib version
Cream background + cardinal red text theme
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import random
import math

random.seed(42)
np.random.seed(42)

# ─── Color Palette (Cream + Cardinal) ───
BG = '#F5F0E8'           # warm cream from user's image
PANEL_BG = '#EDE8DF'     # slightly darker cream for panels
CARDINAL = '#C41E3A'     # cardinal red — main text
CARDINAL_DIM = '#9A1830' # darker cardinal for subtitles
TEXT_DARK = '#3A2A2A'     # dark warm brown for stats
TEXT_MID = '#7A6A60'      # medium brown for secondary text
TEXT_FAINT = '#B0A498'    # faint warm gray
GRID_LINE = '#D8D0C4'    # subtle grid/border
REAL_PKT = '#C41E3A'     # cardinal for real packets
DUMMY_PKT = '#2A7B9B'    # teal-blue for dummy (good contrast on cream)
ACCENT_GREEN = '#2D8C5A' # muted green
ACCENT_TEAL = '#2A7B9B'
ACCENT_PURPLE = '#7B3FA0'
ACCENT_ORANGE = '#D4722A'
ACCENT_DARK = '#8B1A2B'  # darker red

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': PANEL_BG,
    'axes.edgecolor': GRID_LINE,
    'text.color': CARDINAL,
    'xtick.color': TEXT_MID,
    'ytick.color': TEXT_MID,
    'font.family': 'Avenir Next',
    'font.size': 11,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
})


# ═══════════════════════════════════════════════════════════════
# GRAPHIC 1: Defense Spectrum
# ═══════════════════════════════════════════════════════════════

def create_spectrum():
    fig, ax = plt.subplots(figsize=(16, 6.2), dpi=150)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1, 1)
    ax.axis('off')

    fig.suptitle('WEBSITE FINGERPRINTING DEFENSE SPECTRUM',
                 fontsize=24, fontweight='bold', color=CARDINAL, y=0.95)

    # Gradient axis bar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    cmap = LinearSegmentedColormap.from_list('spec', [ACCENT_GREEN, '#B8A030', ACCENT_ORANGE, CARDINAL])
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, -0.04, 0.04], zorder=2)

    # Arrow tips
    ax.annotate('', xy=(-0.02, 0), xytext=(0.02, 0),
                arrowprops=dict(arrowstyle='->', color=ACCENT_GREEN, lw=2))
    ax.annotate('', xy=(1.02, 0), xytext=(0.98, 0),
                arrowprops=dict(arrowstyle='->', color=CARDINAL, lw=2))

    # Axis end labels
    ax.text(0.0, -0.14, 'LOW OVERHEAD', ha='center', fontsize=11,
            fontweight='bold', color=ACCENT_GREEN)
    ax.text(1.0, -0.14, 'HIGH OVERHEAD', ha='center', fontsize=11,
            fontweight='bold', color=CARDINAL)
    ax.text(0.0, -0.22, 'Targeted DL Defense', ha='center', fontsize=9, color=TEXT_FAINT)
    ax.text(1.0, -0.22, 'Strong vs All Attacks', ha='center', fontsize=9, color=TEXT_FAINT)

    # ─── Defense nodes ───
    defenses = [
        {'name': 'BRO', 'pos': 0.06, 'year': '2024', 'color': ACCENT_GREEN,
         'stats': ['Zero delay', 'Low BW overhead', 'Targets DL attacks']},
        {'name': 'WTF-PAD', 'pos': 0.24, 'year': '2016', 'color': ACCENT_TEAL,
         'stats': ['~54% BW', 'Zero latency', 'Adaptive padding']},
        {'name': 'RegulaTor', 'pos': 0.42, 'year': '2022', 'color': '#2080A0',
         'stats': ['Low BW', '~6.6% latency', 'Rate regulation']},
        {'name': 'WalkieTalkie', 'pos': 0.60, 'year': '2017', 'color': ACCENT_PURPLE,
         'stats': ['~31% BW', '~34% latency', 'Half-duplex bursts']},
        {'name': 'Tamaraw', 'pos': 0.80, 'year': '2014', 'color': ACCENT_ORANGE,
         'stats': ['~128% BW', '145\u2013200% latency', 'Dual-rate constant']},
        {'name': 'BuFLO', 'pos': 0.95, 'year': '2012', 'color': CARDINAL,
         'stats': ['>100% BW', '2\u20133\u00d7 latency', 'Constant rate']},
    ]

    for i, d in enumerate(defenses):
        x = d['pos']
        above = (i % 2 == 0)
        sign = 1 if above else -1

        # Node dot on axis
        ax.plot(x, 0, 'o', color=d['color'], markersize=12, zorder=5)
        ax.plot(x, 0, 'o', color=BG, markersize=4, zorder=6)

        # Dashed connector
        stem_end = sign * 0.55
        ax.plot([x, x], [sign * 0.06, stem_end], '--', color=d['color'],
                linewidth=1, alpha=0.6, zorder=3)

        # Defense name
        name_y = stem_end + sign * 0.02
        ax.text(x, name_y, d['name'], ha='center', va='bottom' if above else 'top',
                fontsize=18, fontweight='bold', color=d['color'])

        # Year
        year_y = name_y + sign * 0.10
        ax.text(x, year_y, d['year'], ha='center', va='bottom' if above else 'top',
                fontsize=9, color=TEXT_FAINT)

        # Stats
        for j, stat in enumerate(d['stats']):
            sy = year_y + sign * (0.08 + j * 0.07)
            ax.text(x, sy, stat, ha='center', va='bottom' if above else 'top',
                    fontsize=10, color=TEXT_MID)

    # Bottom line
    ax.axhline(y=-0.90, xmin=0.03, xmax=0.97, color=GRID_LINE, linewidth=1)
    ax.text(0.5, -0.95, 'OVERHEAD  -->', ha='center', fontsize=8, color=TEXT_FAINT,
            family='monospace')

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig('/Users/yousefh/.21st/worktrees/df-website-fingerprinting/yearning-harbor/option1-spectrum-plt.png',
                dpi=150, facecolor=BG, bbox_inches='tight', pad_inches=0.3)
    print('Saved option1-spectrum-plt.png')
    plt.close()


# ═══════════════════════════════════════════════════════════════
# GRAPHIC 2: Traffic Reshaping Panels
# ═══════════════════════════════════════════════════════════════

def create_traffic():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=150)
    fig.suptitle('HOW EACH DEFENSE RESHAPES TRAFFIC',
                 fontsize=24, fontweight='bold', color=CARDINAL, y=0.97)

    # Legend
    real_patch = mpatches.Patch(facecolor=REAL_PKT, label='Real Packets')
    dummy_patch = mpatches.Patch(facecolor=DUMMY_PKT, alpha=0.7, hatch='///',
                                  edgecolor=DUMMY_PKT, label='Dummy / Padding')
    fig.legend(handles=[real_patch, dummy_patch], loc='upper center',
               ncol=2, fontsize=12, frameon=False,
               labelcolor=TEXT_DARK,
               bbox_to_anchor=(0.5, 0.938))

    panels = [
        ('WTF-PAD', 'Fills timing gaps with dummy packets'),
        ('WalkieTalkie', 'Half-duplex constant-rate bursts'),
        ('RegulaTor', 'Smooths rate toward target'),
        ('BRO', 'Front-loads padding where DL looks'),
        ('BuFLO', 'Constant rate, fixed duration'),
        ('Tamaraw', 'Dual-rate constant (in / out)'),
    ]

    for idx, (name, desc) in enumerate(panels):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        ax.set_facecolor(PANEL_BG)
        # Title with extra padding — no subtitle overlapping
        ax.set_title(f'{name}\n', fontsize=17, fontweight='bold', color=CARDINAL, pad=14)
        # Description below title, using figure transform trick
        ax.text(0.5, 1.02, desc, transform=ax.transAxes, ha='center',
                fontsize=9, color=TEXT_FAINT, va='bottom')

        ax.set_xlim(-1, 61)
        ax.set_ylim(-1.15, 1.15)
        ax.axhline(y=0, color=TEXT_FAINT, linewidth=0.5, alpha=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.97, 0.02, 'time -->', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=7, color=TEXT_FAINT, family='monospace')

        for spine in ax.spines.values():
            spine.set_color(GRID_LINE)
            spine.set_linewidth(0.8)

        bw = 0.7

        if name == 'WTF-PAD':
            real_pos = []
            cursor = 0
            while cursor < 60:
                burst = random.randint(3, 7)
                for b in range(burst):
                    if cursor + b < 60:
                        real_pos.append(cursor + b)
                cursor += burst + random.randint(5, 12)
            real_set = set(real_pos)

            for i in range(60):
                if i in real_set:
                    ax.bar(i, random.uniform(0.3, 0.95), width=bw, color=REAL_PKT)
                    ax.bar(i, -random.uniform(0.2, 0.7), width=bw, color=REAL_PKT)
                else:
                    ax.bar(i, random.uniform(0.1, 0.4), width=bw, color=DUMMY_PKT, alpha=0.7,
                           edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)
                    ax.bar(i, -random.uniform(0.08, 0.3), width=bw, color=DUMMY_PKT, alpha=0.7,
                           edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)

        elif name == 'WalkieTalkie':
            x = 0
            for burst_idx in range(6):
                is_client = (burst_idx % 2 == 0)
                n_bars = 7
                uniform_h = 0.7
                sign = 1 if is_client else -1

                for b in range(n_bars):
                    real_h = random.uniform(0.25, 0.6)
                    pad_h = uniform_h - real_h
                    ax.bar(x, sign * real_h, width=bw, color=REAL_PKT)
                    ax.bar(x, sign * pad_h, bottom=sign * real_h, width=bw,
                           color=DUMMY_PKT, alpha=0.7, edgecolor=DUMMY_PKT,
                           hatch='///', linewidth=0.3)
                    x += 1
                x += 1.5

        elif name == 'RegulaTor':
            target_up = 0.45
            target_dn = 0.40
            ax.axhline(y=target_up, color=ACCENT_GREEN, linewidth=0.8, alpha=0.4, linestyle='--')
            ax.axhline(y=-target_dn, color=ACCENT_GREEN, linewidth=0.8, alpha=0.4, linestyle='--')
            ax.text(59, target_up + 0.06, 'target', fontsize=7, color=ACCENT_GREEN, alpha=0.7, ha='right')
            ax.text(59, -target_dn - 0.12, 'target', fontsize=7, color=ACCENT_GREEN, alpha=0.7, ha='right')

            for i in range(55):
                h_up = target_up + random.uniform(-0.08, 0.08)
                h_dn = target_dn + random.uniform(-0.07, 0.07)
                real_up = random.uniform(h_up * 0.5, h_up)
                real_dn = random.uniform(h_dn * 0.5, h_dn)

                ax.bar(i, real_up, width=bw, color=REAL_PKT)
                ax.bar(i, h_up - real_up, bottom=real_up, width=bw,
                       color=DUMMY_PKT, alpha=0.7, edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)
                ax.bar(i, -real_dn, width=bw, color=REAL_PKT)
                ax.bar(i, -(h_dn - real_dn), bottom=-real_dn, width=bw,
                       color=DUMMY_PKT, alpha=0.7, edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)

        elif name == 'BRO':
            for i in range(55):
                t = i / 55.0
                if random.random() < 0.6:
                    ax.bar(i, random.uniform(0.2, 0.75), width=bw, color=REAL_PKT)
                    ax.bar(i, -random.uniform(0.15, 0.55), width=bw, color=REAL_PKT)

                pad_int = math.exp(-5.0 * t)
                if pad_int > 0.05:
                    ph_up = 0.8 * pad_int + random.uniform(0, 0.08)
                    ph_dn = 0.6 * pad_int + random.uniform(0, 0.06)
                    ax.bar(i + 0.4, ph_up, width=bw * 0.6, color=DUMMY_PKT, alpha=0.7,
                           edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)
                    ax.bar(i + 0.4, -ph_dn, width=bw * 0.6, color=DUMMY_PKT, alpha=0.7,
                           edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)

            # Bracket below the bars instead of above (avoid title overlap)
            ax.annotate('', xy=(0, -1.0), xytext=(14, -1.0),
                        arrowprops=dict(arrowstyle='|-|', color=DUMMY_PKT, lw=1.2))
            ax.text(7, -1.08, 'dense padding zone', ha='center', fontsize=7,
                    color=DUMMY_PKT, alpha=0.8)

        elif name == 'BuFLO':
            constant_h = 0.55
            ax.axhline(y=constant_h, color=TEXT_FAINT, linewidth=0.8, alpha=0.4, linestyle=':')
            ax.axhline(y=-constant_h, color=TEXT_FAINT, linewidth=0.8, alpha=0.4, linestyle=':')

            for i in range(50):
                is_real = random.random() < 0.35
                if is_real:
                    rh = random.uniform(0.2, constant_h)
                    ax.bar(i, rh, width=bw, color=REAL_PKT)
                    ax.bar(i, constant_h - rh, bottom=rh, width=bw,
                           color=DUMMY_PKT, alpha=0.7, edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)
                    rh_d = random.uniform(0.15, constant_h)
                    ax.bar(i, -rh_d, width=bw, color=REAL_PKT)
                    ax.bar(i, -(constant_h - rh_d), bottom=-rh_d, width=bw,
                           color=DUMMY_PKT, alpha=0.7, edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)
                else:
                    ax.bar(i, constant_h, width=bw, color=DUMMY_PKT, alpha=0.7,
                           edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)
                    ax.bar(i, -constant_h, width=bw, color=DUMMY_PKT, alpha=0.7,
                           edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)

        elif name == 'Tamaraw':
            fast_h = 0.6
            slow_h = 0.45
            ax.axhline(y=fast_h, color=TEXT_FAINT, linewidth=0.6, alpha=0.3, linestyle=':')
            ax.axhline(y=-slow_h, color=TEXT_FAINT, linewidth=0.6, alpha=0.3, linestyle=':')
            ax.text(1, fast_h + 0.06, 'p_in (fast)', fontsize=7, color=TEXT_MID)
            ax.text(1, -slow_h - 0.12, 'p_out (slow)', fontsize=7, color=TEXT_MID)

            for i in range(55):
                is_real = random.random() < 0.35
                if is_real:
                    rh = random.uniform(0.25, fast_h)
                    ax.bar(i, rh, width=bw * 0.7, color=REAL_PKT)
                    if fast_h - rh > 0.02:
                        ax.bar(i, fast_h - rh, bottom=rh, width=bw * 0.7,
                               color=DUMMY_PKT, alpha=0.7, edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)
                else:
                    ax.bar(i, fast_h, width=bw * 0.7, color=DUMMY_PKT, alpha=0.7,
                           edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)

            for i in range(0, 55, 3):
                is_real = random.random() < 0.4
                if is_real:
                    rh = random.uniform(0.2, slow_h)
                    ax.bar(i, -rh, width=bw * 0.9, color=REAL_PKT)
                    if slow_h - rh > 0.02:
                        ax.bar(i, -(slow_h - rh), bottom=-rh, width=bw * 0.9,
                               color=DUMMY_PKT, alpha=0.7, edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)
                else:
                    ax.bar(i, -slow_h, width=bw * 0.9, color=DUMMY_PKT, alpha=0.7,
                           edgecolor=DUMMY_PKT, hatch='///', linewidth=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.subplots_adjust(hspace=0.55, wspace=0.15)
    fig.savefig('/Users/yousefh/.21st/worktrees/df-website-fingerprinting/yearning-harbor/option2-traffic-plt.png',
                dpi=150, facecolor=BG, bbox_inches='tight', pad_inches=0.4)
    print('Saved option2-traffic-plt.png')
    plt.close()


if __name__ == '__main__':
    create_spectrum()
    create_traffic()
    print('\nDone.')
