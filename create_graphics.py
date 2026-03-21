#!/usr/bin/env python3
"""
Signal Cartography — WF Defense Visualizations
Two presentation-ready graphics for website fingerprinting defenses.
"""

from PIL import Image, ImageDraw, ImageFont
import math
import random

random.seed(42)

# ─── Color Palette (Signal Cartography) ───
BG = (13, 15, 22)
BG_PANEL = (19, 22, 32)
AXIS_DARK = (28, 32, 48)
CYAN = (0, 200, 220)
CYAN_DIM = (0, 120, 140)
CYAN_GLOW = (0, 255, 255)
WARM = (255, 140, 50)
WARM_DIM = (180, 90, 30)
WARM_GLOW = (255, 170, 80)
WHITE = (230, 232, 240)
WHITE_DIM = (140, 145, 165)
WHITE_FAINT = (70, 75, 95)
GRID_LINE = (30, 35, 55)
ACCENT_GREEN = (60, 220, 140)
ACCENT_RED = (255, 80, 90)
ACCENT_PURPLE = (160, 100, 255)
REAL_PKT = (0, 190, 220)       # cyan for real packets
DUMMY_PKT = (255, 140, 50)     # warm orange for dummy/padding
REAL_PKT_ALT = (80, 200, 180)  # teal variant


def load_fonts():
    """Load system fonts with fallbacks."""
    fonts = {}

    # Try font paths in order of preference
    font_paths = {
        'light': [
            '/System/Library/Fonts/Supplemental/Avenir Next.ttc',
            '/System/Library/Fonts/Helvetica.ttc',
            '/System/Library/Fonts/SFNSDisplay.ttf',
        ],
        'bold': [
            '/System/Library/Fonts/Supplemental/Avenir Next.ttc',
            '/System/Library/Fonts/Helvetica.ttc',
        ],
        'mono': [
            '/System/Library/Fonts/SFNSMono.ttf',
            '/System/Library/Fonts/Menlo.ttc',
            '/System/Library/Fonts/Monaco.ttf',
        ]
    }

    def try_load(paths, size, index=0):
        for p in paths:
            try:
                return ImageFont.truetype(p, size, index=index)
            except (OSError, IOError):
                try:
                    return ImageFont.truetype(p, size)
                except (OSError, IOError):
                    continue
        return ImageFont.load_default()

    fonts['title'] = try_load(font_paths['bold'], 28, index=1)
    fonts['subtitle'] = try_load(font_paths['light'], 18)
    fonts['defense'] = try_load(font_paths['bold'], 22, index=1)
    fonts['defense_lg'] = try_load(font_paths['bold'], 26, index=1)
    fonts['stat'] = try_load(font_paths['light'], 13)
    fonts['stat_sm'] = try_load(font_paths['light'], 11)
    fonts['label'] = try_load(font_paths['light'], 14)
    fonts['label_sm'] = try_load(font_paths['light'], 12)
    fonts['axis'] = try_load(font_paths['light'], 15)
    fonts['panel_title'] = try_load(font_paths['bold'], 20, index=1)
    fonts['legend'] = try_load(font_paths['light'], 13)
    fonts['tiny'] = try_load(font_paths['mono'], 10)
    fonts['header'] = try_load(font_paths['bold'], 34, index=1)

    return fonts


def draw_glow_line(draw, points, color, width=2, glow_radius=4):
    """Draw a line with a subtle glow effect."""
    r, g, b = color
    for i in range(glow_radius, 0, -1):
        alpha_factor = 0.15 * (1 - i / glow_radius)
        gc = (int(r * alpha_factor), int(g * alpha_factor), int(b * alpha_factor))
        blended = (
            min(255, gc[0] + BG[0]),
            min(255, gc[1] + BG[1]),
            min(255, gc[2] + BG[2])
        )
        draw.line(points, fill=blended, width=width + i * 2)
    draw.line(points, fill=color, width=width)


def draw_circle_node(draw, cx, cy, radius, color, glow=True):
    """Draw a circle with optional glow."""
    r, g, b = color
    if glow:
        for i in range(12, 0, -1):
            alpha_factor = 0.08 * (1 - i / 12)
            gc = (
                min(255, int(r * alpha_factor) + BG[0]),
                min(255, int(g * alpha_factor) + BG[1]),
                min(255, int(b * alpha_factor) + BG[2])
            )
            draw.ellipse(
                [cx - radius - i * 2, cy - radius - i * 2,
                 cx + radius + i * 2, cy + radius + i * 2],
                fill=gc
            )
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=color, outline=color
    )


def text_center(draw, x, y, text, font, fill):
    """Draw text centered at (x, y)."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x - tw // 2, y - th // 2), text, font=font, fill=fill)


# ═══════════════════════════════════════════════════════════════
# GRAPHIC 1: Defense Spectrum
# ═══════════════════════════════════════════════════════════════

def create_spectrum_graphic(fonts):
    W, H = 1600, 620
    img = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Subtle grid lines
    for x in range(0, W, 40):
        draw.line([(x, 0), (x, H)], fill=(18, 21, 30), width=1)
    for y in range(0, H, 40):
        draw.line([(0, y), (W, y)], fill=(18, 21, 30), width=1)

    # Title area
    text_center(draw, W // 2, 36, "WEBSITE FINGERPRINTING DEFENSE SPECTRUM", fonts['header'], WHITE)

    # Axis parameters
    axis_y = 310
    ax_left = 120
    ax_right = W - 120
    ax_width = ax_right - ax_left

    # Draw gradient axis band
    band_h = 8
    for x in range(ax_left, ax_right):
        t = (x - ax_left) / ax_width
        r = int(ACCENT_GREEN[0] * (1 - t) + ACCENT_RED[0] * t)
        g = int(ACCENT_GREEN[1] * (1 - t) + ACCENT_RED[1] * t)
        b = int(ACCENT_GREEN[2] * (1 - t) + ACCENT_RED[2] * t)
        draw.line([(x, axis_y - band_h // 2), (x, axis_y + band_h // 2)], fill=(r, g, b))

    # Glow around axis
    for i in range(8, 0, -1):
        for x in range(ax_left, ax_right, 3):
            t = (x - ax_left) / ax_width
            r = int(ACCENT_GREEN[0] * (1 - t) + ACCENT_RED[0] * t)
            g = int(ACCENT_GREEN[1] * (1 - t) + ACCENT_RED[1] * t)
            b = int(ACCENT_GREEN[2] * (1 - t) + ACCENT_RED[2] * t)
            alpha = 0.04 * (1 - i / 8)
            gc = (
                min(255, int(r * alpha) + BG[0]),
                min(255, int(g * alpha) + BG[1]),
                min(255, int(b * alpha) + BG[2])
            )
            draw.line([(x, axis_y - band_h // 2 - i * 3), (x, axis_y - band_h // 2 - i * 3 + 1)], fill=gc)
            draw.line([(x, axis_y + band_h // 2 + i * 3), (x, axis_y + band_h // 2 + i * 3 + 1)], fill=gc)

    # Arrow endpoints
    arrow_size = 12
    # Left arrow
    draw.polygon([
        (ax_left - 2, axis_y),
        (ax_left + arrow_size, axis_y - arrow_size),
        (ax_left + arrow_size, axis_y + arrow_size)
    ], fill=ACCENT_GREEN)
    # Right arrow
    draw.polygon([
        (ax_right + 2, axis_y),
        (ax_right - arrow_size, axis_y - arrow_size),
        (ax_right - arrow_size, axis_y + arrow_size)
    ], fill=ACCENT_RED)

    # Axis labels
    text_center(draw, ax_left + 40, axis_y + 36, "LOW OVERHEAD", fonts['label'], ACCENT_GREEN)
    text_center(draw, ax_right - 40, axis_y + 36, "HIGH OVERHEAD", fonts['label'], ACCENT_RED)

    # Secondary annotation line
    text_center(draw, ax_left + 80, axis_y + 58, "Targeted DL Defense", fonts['stat_sm'], WHITE_FAINT)
    text_center(draw, ax_right - 80, axis_y + 58, "Strong vs All Attacks", fonts['stat_sm'], WHITE_FAINT)

    # ─── Defense Nodes ───
    defenses = [
        {
            'name': 'BRO',
            'pos': 0.06,
            'stats': ['Zero delay', 'Low BW overhead', 'Targets DL attacks'],
            'year': '2024',
            'color': ACCENT_GREEN,
        },
        {
            'name': 'WTF-PAD',
            'pos': 0.22,
            'stats': ['~54% BW', 'Zero latency', 'Adaptive padding'],
            'year': '2016',
            'color': (40, 200, 170),
        },
        {
            'name': 'RegulaTor',
            'pos': 0.40,
            'stats': ['Low BW', '~6.6% latency', 'Rate regulation'],
            'year': '2022',
            'color': CYAN,
        },
        {
            'name': 'WalkieTalkie',
            'pos': 0.58,
            'stats': ['~31% BW', '~34% latency', 'Half-duplex bursts'],
            'year': '2017',
            'color': ACCENT_PURPLE,
        },
        {
            'name': 'Tamaraw',
            'pos': 0.78,
            'stats': ['~128% BW', '145–200% latency', 'Dual-rate constant'],
            'year': '2014',
            'color': WARM,
        },
        {
            'name': 'BuFLO',
            'pos': 0.94,
            'stats': ['>100% BW', '2–3× latency', 'Constant rate'],
            'year': '2012',
            'color': ACCENT_RED,
        },
    ]

    for i, d in enumerate(defenses):
        x = int(ax_left + d['pos'] * ax_width)

        # Alternating above/below for visual clarity
        above = (i % 2 == 0)

        # Vertical connector line
        node_y = axis_y
        if above:
            stem_top = axis_y - 130
            stem_bot = axis_y - band_h // 2 - 2
            text_anchor_y = stem_top - 10
        else:
            stem_top = axis_y + band_h // 2 + 2
            stem_bot = axis_y + 130
            text_anchor_y = stem_bot + 10

        # Dashed connector
        dash_len = 6
        gap = 4
        if above:
            cy = stem_bot
            while cy > stem_top:
                draw.line([(x, cy), (x, max(stem_top, cy - dash_len))], fill=d['color'], width=1)
                cy -= (dash_len + gap)
        else:
            cy = stem_top
            while cy < stem_bot:
                draw.line([(x, cy), (x, min(stem_bot, cy + dash_len))], fill=d['color'], width=1)
                cy += (dash_len + gap)

        # Node dot on axis
        draw_circle_node(draw, x, node_y, 7, d['color'], glow=True)
        draw_circle_node(draw, x, node_y, 3, WHITE, glow=False)

        # Defense name
        if above:
            name_y = text_anchor_y - 50
        else:
            name_y = text_anchor_y + 4

        text_center(draw, x, name_y, d['name'], fonts['defense_lg'], d['color'])

        # Year tag
        year_y = name_y + 24
        text_center(draw, x, year_y, d['year'], fonts['stat_sm'], WHITE_FAINT)

        # Stats
        for j, stat in enumerate(d['stats']):
            stat_y = year_y + 18 + j * 16
            text_center(draw, x, stat_y, stat, fonts['stat'], WHITE_DIM)

    # Bottom decorative line
    draw.line([(60, H - 30), (W - 60, H - 30)], fill=GRID_LINE, width=1)
    text_center(draw, W // 2, H - 16, "OVERHEAD  →", fonts['tiny'], WHITE_FAINT)

    img.save('/Users/yousefh/.21st/worktrees/df-website-fingerprinting/yearning-harbor/option1-spectrum.png', dpi=(144, 144))
    print("✓ option1-spectrum.png saved")
    return img


# ═══════════════════════════════════════════════════════════════
# GRAPHIC 2: Traffic Reshaping Panels
# ═══════════════════════════════════════════════════════════════

def draw_packet_bar(draw, x, y_base, height, width, color, dashed=False):
    """Draw a single packet bar (up or down from baseline)."""
    if dashed:
        dash = 3
        gap = 3
        if height > 0:  # going up (negative y direction)
            cy = y_base
            target = y_base - height
            while cy > target:
                end = max(target, cy - dash)
                draw.rectangle([x, end, x + width, cy], fill=color)
                cy -= (dash + gap)
        else:  # going down
            cy = y_base
            target = y_base - height  # height is negative so target > y_base
            while cy < target:
                end = min(target, cy + dash)
                draw.rectangle([x, cy, x + width, end], fill=color)
                cy += (dash + gap)
    else:
        if height > 0:
            draw.rectangle([x, y_base - height, x + width, y_base], fill=color)
        elif height < 0:
            draw.rectangle([x, y_base, x + width, y_base - height], fill=color)


def create_traffic_graphic(fonts):
    W, H = 1600, 940
    img = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Title
    text_center(draw, W // 2, 30, "HOW EACH DEFENSE RESHAPES TRAFFIC", fonts['header'], WHITE)

    # Legend
    legend_y = 62
    legend_x = W // 2 - 180
    # Real packet swatch
    draw.rectangle([legend_x, legend_y, legend_x + 30, legend_y + 10], fill=REAL_PKT)
    draw.text((legend_x + 38, legend_y - 2), "Real Packets", font=fonts['legend'], fill=WHITE_DIM)
    # Dummy packet swatch (dashed look)
    dx = legend_x + 200
    for i in range(0, 30, 6):
        draw.rectangle([dx + i, legend_y, dx + i + 3, legend_y + 10], fill=DUMMY_PKT)
    draw.text((dx + 38, legend_y - 2), "Dummy / Padding", font=fonts['legend'], fill=WHITE_DIM)

    # Panel grid: 3 columns × 2 rows
    cols, rows = 3, 2
    margin_x, margin_y = 50, 90
    gap_x, gap_y = 24, 24
    panel_w = (W - 2 * margin_x - (cols - 1) * gap_x) // cols
    panel_h = (H - margin_y - 60 - (rows - 1) * gap_y) // rows

    panels = [
        ('WTF-PAD', 'Fills timing gaps with dummy packets'),
        ('WalkieTalkie', 'Half-duplex constant-rate bursts'),
        ('RegulaTor', 'Smooths rate toward target'),
        ('BRO', 'Front-loads padding where DL looks'),
        ('BuFLO', 'Constant rate, fixed duration'),
        ('Tamaraw', 'Dual-rate constant (in / out)'),
    ]

    for idx, (name, desc) in enumerate(panels):
        col = idx % cols
        row = idx // cols
        px = margin_x + col * (panel_w + gap_x)
        py = margin_y + row * (panel_h + gap_y)

        # Panel background
        draw.rounded_rectangle([px, py, px + panel_w, py + panel_h], radius=6, fill=BG_PANEL)
        draw.rounded_rectangle([px, py, px + panel_w, py + panel_h], radius=6, outline=GRID_LINE, width=1)

        # Panel title
        text_center(draw, px + panel_w // 2, py + 20, name, fonts['panel_title'], CYAN)
        draw.text((px + 14, py + 38), desc, font=fonts['stat_sm'], fill=WHITE_FAINT)

        # Drawing area within panel
        area_left = px + 20
        area_right = px + panel_w - 20
        area_top = py + 60
        area_bot = py + panel_h - 20
        area_w = area_right - area_left
        area_h = area_bot - area_top
        baseline = area_top + area_h // 2

        # Faint baseline
        draw.line([(area_left, baseline), (area_right, baseline)], fill=WHITE_FAINT, width=1)

        # Faint "time →" label
        draw.text((area_right - 40, area_bot - 12), "time →", font=fonts['tiny'], fill=(50, 55, 75))

        bar_w = 3
        max_h = (area_h // 2) - 10

        if name == 'WTF-PAD':
            # Irregular real traffic with dummy packets filling gaps
            # Generate bursty real traffic
            num_bars = 60
            real_positions = []
            # Create bursts
            x_cursor = 0
            while x_cursor < num_bars:
                burst_len = random.randint(3, 8)
                for b in range(burst_len):
                    if x_cursor + b < num_bars:
                        real_positions.append(x_cursor + b)
                gap = random.randint(6, 14)
                x_cursor += burst_len + gap

            spacing = area_w / (num_bars + 1)
            for i in range(num_bars):
                bx = int(area_left + (i + 1) * spacing)
                if i in real_positions:
                    h_up = random.randint(int(max_h * 0.3), int(max_h * 0.9))
                    h_down = random.randint(int(max_h * 0.2), int(max_h * 0.7))
                    draw_packet_bar(draw, bx, baseline, h_up, bar_w, REAL_PKT)
                    draw_packet_bar(draw, bx, baseline, -h_down, bar_w, REAL_PKT)
                else:
                    # Dummy packets in gaps (smaller, dashed)
                    h_up = random.randint(int(max_h * 0.15), int(max_h * 0.45))
                    h_down = random.randint(int(max_h * 0.1), int(max_h * 0.35))
                    draw_packet_bar(draw, bx, baseline, h_up, bar_w, DUMMY_PKT, dashed=True)
                    draw_packet_bar(draw, bx, baseline, -h_down, bar_w, DUMMY_PKT, dashed=True)

        elif name == 'WalkieTalkie':
            # Alternating half-duplex bursts, padded to equal sizes
            burst_count = 6
            bars_per_burst = 7
            total_bars = burst_count * bars_per_burst
            spacing = area_w / (total_bars + burst_count * 2)
            bx = area_left + 10

            for burst_idx in range(burst_count):
                is_client = (burst_idx % 2 == 0)
                uniform_h = int(max_h * 0.7)

                for b in range(bars_per_burst):
                    if is_client:
                        # Outgoing (up) — some real, some padded to uniform
                        real_h = random.randint(int(max_h * 0.3), int(max_h * 0.65))
                        draw_packet_bar(draw, int(bx), baseline, real_h, bar_w, REAL_PKT)
                        # Padding on top
                        pad_h = uniform_h - real_h
                        if pad_h > 2:
                            draw_packet_bar(draw, int(bx), baseline - real_h, pad_h, bar_w, DUMMY_PKT, dashed=True)
                    else:
                        # Incoming (down)
                        real_h = random.randint(int(max_h * 0.3), int(max_h * 0.65))
                        draw_packet_bar(draw, int(bx), baseline, -real_h, bar_w, REAL_PKT)
                        pad_h = uniform_h - real_h
                        if pad_h > 2:
                            draw_packet_bar(draw, int(bx), baseline + real_h, -pad_h, bar_w, DUMMY_PKT, dashed=True)

                    bx += spacing

                # Gap between bursts
                bx += spacing * 1.5

        elif name == 'RegulaTor':
            # Smoothed rate-limited version: original spiky → regulated
            num_bars = 55
            spacing = area_w / (num_bars + 1)

            # Target rate lines
            target_up = int(max_h * 0.45)
            target_down = int(max_h * 0.40)

            # Draw faint target rate lines
            draw.line([(area_left, baseline - target_up), (area_right, baseline - target_up)],
                      fill=(50, 70, 60), width=1)
            draw.line([(area_left, baseline + target_down), (area_right, baseline + target_down)],
                      fill=(50, 70, 60), width=1)
            draw.text((area_right - 65, baseline - target_up - 12), "target ↑", font=fonts['tiny'], fill=(60, 80, 70))
            draw.text((area_right - 65, baseline + target_down + 2), "target ↓", font=fonts['tiny'], fill=(60, 80, 70))

            for i in range(num_bars):
                bx = int(area_left + (i + 1) * spacing)
                # Regulated: heights cluster near target with small variance
                h_up = target_up + random.randint(-12, 12)
                h_down = target_down + random.randint(-10, 10)

                # Some are real, some padded
                real_up = random.randint(int(h_up * 0.5), h_up)
                real_down = random.randint(int(h_down * 0.5), h_down)

                draw_packet_bar(draw, bx, baseline, real_up, bar_w, REAL_PKT)
                if h_up - real_up > 2:
                    draw_packet_bar(draw, bx, baseline - real_up, h_up - real_up, bar_w, DUMMY_PKT, dashed=True)

                draw_packet_bar(draw, bx, baseline, -real_down, bar_w, REAL_PKT)
                if h_down - real_down > 2:
                    draw_packet_bar(draw, bx, baseline + real_down, -(h_down - real_down), bar_w, DUMMY_PKT, dashed=True)

        elif name == 'BRO':
            # Heavy padding at START, tapering off
            num_bars = 55
            spacing = area_w / (num_bars + 1)

            for i in range(num_bars):
                bx = int(area_left + (i + 1) * spacing)
                t = i / num_bars  # 0 = start, 1 = end

                # Real traffic: bursty throughout
                if random.random() < 0.6:
                    real_h_up = random.randint(int(max_h * 0.2), int(max_h * 0.7))
                    real_h_down = random.randint(int(max_h * 0.15), int(max_h * 0.5))
                    draw_packet_bar(draw, bx, baseline, real_h_up, bar_w, REAL_PKT)
                    draw_packet_bar(draw, bx, baseline, -real_h_down, bar_w, REAL_PKT)

                # Padding: heavy at start, exponential decay
                pad_intensity = math.exp(-4.0 * t)  # exponential decay from start
                if pad_intensity > 0.05:
                    pad_h_up = int(max_h * 0.8 * pad_intensity) + random.randint(0, 10)
                    pad_h_down = int(max_h * 0.6 * pad_intensity) + random.randint(0, 8)
                    draw_packet_bar(draw, bx + bar_w + 1, baseline, pad_h_up, bar_w, DUMMY_PKT, dashed=True)
                    draw_packet_bar(draw, bx + bar_w + 1, baseline, -pad_h_down, bar_w, DUMMY_PKT, dashed=True)

            # Annotation: bracket showing "heavy padding zone"
            zone_end_x = int(area_left + 0.25 * area_w)
            bracket_y = area_top + 4
            draw.line([(area_left + 10, bracket_y), (zone_end_x, bracket_y)], fill=WARM_DIM, width=1)
            draw.line([(area_left + 10, bracket_y), (area_left + 10, bracket_y + 6)], fill=WARM_DIM, width=1)
            draw.line([(zone_end_x, bracket_y), (zone_end_x, bracket_y + 6)], fill=WARM_DIM, width=1)
            text_center(draw, (area_left + 10 + zone_end_x) // 2, bracket_y - 10, "dense padding zone", fonts['tiny'], WARM_DIM)

        elif name == 'BuFLO':
            # Perfectly constant rate: uniform bars, equal spacing
            num_bars = 50
            spacing = area_w / (num_bars + 1)
            constant_h = int(max_h * 0.55)

            for i in range(num_bars):
                bx = int(area_left + (i + 1) * spacing)

                # Some bars are real, some are padding, but all same height
                is_real = random.random() < 0.4
                if is_real:
                    real_h = random.randint(int(constant_h * 0.4), constant_h)
                    draw_packet_bar(draw, bx, baseline, real_h, bar_w, REAL_PKT)
                    pad_h = constant_h - real_h
                    if pad_h > 2:
                        draw_packet_bar(draw, bx, baseline - real_h, pad_h, bar_w, DUMMY_PKT, dashed=True)

                    real_h_d = random.randint(int(constant_h * 0.3), constant_h)
                    draw_packet_bar(draw, bx, baseline, -real_h_d, bar_w, REAL_PKT)
                    pad_h_d = constant_h - real_h_d
                    if pad_h_d > 2:
                        draw_packet_bar(draw, bx, baseline + real_h_d, -pad_h_d, bar_w, DUMMY_PKT, dashed=True)
                else:
                    # Pure padding
                    draw_packet_bar(draw, bx, baseline, constant_h, bar_w, DUMMY_PKT, dashed=True)
                    draw_packet_bar(draw, bx, baseline, -constant_h, bar_w, DUMMY_PKT, dashed=True)

            # Constant rate line
            draw.line([(area_left, baseline - constant_h), (area_right, baseline - constant_h)],
                      fill=(80, 85, 105), width=1)
            draw.line([(area_left, baseline + constant_h), (area_right, baseline + constant_h)],
                      fill=(80, 85, 105), width=1)

        elif name == 'Tamaraw':
            # Two separate constant rates: fast incoming, slow outgoing
            fast_spacing = area_w / 56  # dense for incoming
            slow_spacing = area_w / 22  # sparse for outgoing

            fast_h = int(max_h * 0.6)
            slow_h = int(max_h * 0.45)

            # Rate labels
            draw.text((area_left + 2, baseline - fast_h - 14), "ρ_in (fast)", font=fonts['tiny'], fill=WHITE_FAINT)
            draw.text((area_left + 2, baseline + slow_h + 4), "ρ_out (slow)", font=fonts['tiny'], fill=WHITE_FAINT)

            # Rate lines
            draw.line([(area_left, baseline - fast_h), (area_right, baseline - fast_h)],
                      fill=(40, 50, 60), width=1)
            draw.line([(area_left, baseline + slow_h), (area_right, baseline + slow_h)],
                      fill=(40, 50, 60), width=1)

            # Incoming (download, going down from baseline... but let's use UP for visual distinction)
            # Actually: convention — let's do incoming=down, outgoing=up
            # Fast incoming (dense bars going down)
            for i in range(55):
                bx = int(area_left + 8 + i * fast_spacing)
                if bx > area_right - 5:
                    break
                is_real = random.random() < 0.35
                if is_real:
                    rh = random.randint(int(fast_h * 0.4), fast_h)
                    draw_packet_bar(draw, bx, baseline, rh, bar_w, REAL_PKT)
                    if fast_h - rh > 2:
                        draw_packet_bar(draw, bx, baseline - rh, fast_h - rh, bar_w, DUMMY_PKT, dashed=True)
                else:
                    draw_packet_bar(draw, bx, baseline, fast_h, bar_w, DUMMY_PKT, dashed=True)

            # Slow outgoing (sparse bars going down from baseline)
            for i in range(21):
                bx = int(area_left + 8 + i * slow_spacing)
                if bx > area_right - 5:
                    break
                is_real = random.random() < 0.4
                if is_real:
                    rh = random.randint(int(slow_h * 0.4), slow_h)
                    draw_packet_bar(draw, bx + 1, baseline, -rh, bar_w, REAL_PKT)
                    if slow_h - rh > 2:
                        draw_packet_bar(draw, bx + 1, baseline + rh, -(slow_h - rh), bar_w, DUMMY_PKT, dashed=True)
                else:
                    draw_packet_bar(draw, bx + 1, baseline, -slow_h, bar_w, DUMMY_PKT, dashed=True)

    img.save('/Users/yousefh/.21st/worktrees/df-website-fingerprinting/yearning-harbor/option2-traffic.png', dpi=(144, 144))
    print("✓ option2-traffic.png saved")
    return img


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    fonts = load_fonts()
    create_spectrum_graphic(fonts)
    create_traffic_graphic(fonts)
    print("\nDone — both graphics created.")
