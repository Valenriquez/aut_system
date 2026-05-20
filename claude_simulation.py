#!/usr/bin/env python3
"""
claude_simulation.py -- offline visualization of the AlphaBot2 policy_runner.

This mirrors claude_first.py (same 7x7 grid, policy, obstacles, fixed SVG
marker positions and SVG decoding) but, instead of driving the real robot
over ROS 2, it ANIMATES the run with matplotlib:

  * the 7x7 grid with obstacles, start and goal;
  * the optimal policy drawn as arrows;
  * the printable ArUco SVGs rendered -- as their real decoded bit
    patterns -- on the FIXED cells where they are taped (MARKER_TO_CELL);
  * the AlphaBot2 walking the policy path step by step, with a camera
    field-of-view cone; and
  * a "marker recognized" event each time the robot reaches a marker
    cell -- exactly the localization step claude_first.py performs.

No ROS, no hardware. Run:
    python3 claude_simulation.py           # interactive animation
    python3 claude_simulation.py --save    # also write a GIF + PNG
"""
import os
import re
import sys
import xml.etree.ElementTree as ET
from math import atan2, degrees

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Wedge
from matplotlib.animation import FuncAnimation, PillowWriter

# ===================== GRID / WORLD =====================
GRID_SIZE = 7
START = (0, 0)
GOAL  = (6, 6)
OBSTACLES = {
    (1, 0), (1, 2), (1, 3), (1, 4), (1, 6),
    (3, 1), (3, 2), (3, 3), (3, 5),
    (4, 3), (4, 5),
    (5, 1), (5, 5),
    (6, 1), (6, 3), (6, 5),
}

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_DELTA    = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
ACTION_NAME     = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT', -1: '--'}
ACTION_ARROW    = {UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→'}
# unit vectors in plot coords (x = col, y = row, row grows downward)
ACTION_VEC      = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
DELTA_TO_ACTION = {v: k for k, v in ACTION_DELTA.items()}

# ===================== POLICY =====================
# Same hardcoded value-iteration policy as claude_first.py.
# Action codes: UP=0, DOWN=1, LEFT=2, RIGHT=3, obstacle/goal = -1.
#
#         col:  0   1   2   3   4   5   6
policy = np.array([
    [    RIGHT, DOWN, RIGHT, RIGHT, RIGHT, DOWN,  LEFT],   # row 0
    [       -1, DOWN,    -1,    -1,    -1, DOWN,    -1],   # row 1
    [    RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, DOWN],  # row 2
    [       UP,   -1,    -1,    -1,    UP,    -1, DOWN],   # row 3
    [       UP, LEFT,  DOWN,    -1,    UP,    -1, DOWN],   # row 4
    [       UP,   -1, RIGHT, RIGHT,    UP,    -1, DOWN],   # row 5
    [       UP,   -1,    UP,    -1,    UP,    -1,   -1],   # row 6 (goal at 6,6)
], dtype=int)

# ===================== FIXED SVG MARKER POSITIONS =====================
# Directory holding the printable ArUco SVG markers (svgs/4x4_1000-<id>.svg).
SVG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svgs')

# Same FIXED, hand-set table as claude_first.py: the exact cell each SVG
# marker is physically taped onto. Every cell must lie on the policy path.
#
#   ArUco id : (row, col) on the 7x7 grid
MARKER_TO_CELL = {
    0: (0, 0),   # policy-path start
    1: (0, 1),   # turn: RIGHT -> DOWN
    2: (2, 1),   # turn: DOWN  -> RIGHT
    3: (2, 3),   # middle of the row-2 corridor
    4: (2, 6),   # turn: RIGHT -> DOWN
    5: (4, 6),   # middle of the column-6 descent
    6: (6, 6),   # policy-path goal
}

# ===================== SVG HELPERS (same as claude_first.py) =====================

def load_svg_markers(svg_dir):
    """Scan `svg_dir` for ArUco SVGs and return (markers, aruco_dict_name).

    `markers` is a list of (marker_id, file_path) sorted by id.
    """
    markers = []
    dict_name = 'DICT_4X4_1000'
    if not os.path.isdir(svg_dir):
        return markers, dict_name

    pat = re.compile(r'(\d+)x(\d+)_(\d+)[-_](\d+)\.svg$', re.IGNORECASE)
    for path in sorted(__import__('glob').glob(os.path.join(svg_dir, '*.svg'))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        bw, bh, size, marker_id = m.groups()
        dict_name = f'DICT_{bw}X{bh}_{size}'
        markers.append((int(marker_id), path))

    markers.sort(key=lambda t: t[0])
    return markers, dict_name


def decode_svg_grid(svg_path):
    """Parse an ArUco SVG into a 6x6 bit grid (1 = white cell, 0 = black).

    The SVGs draw white <rect> cells over a 6x6 black canvas. A cell is
    white when its centre falls inside any white rect.
    """
    with open(svg_path, 'r') as f:
        text = f.read()
    text = re.sub(r'\sxmlns="[^"]+"', '', text, count=1)
    root = ET.fromstring(text)

    white = []
    for el in root.iter('rect'):
        if (el.get('fill') or '').lower() != 'white':
            continue
        white.append((
            float(el.get('x', 0)), float(el.get('y', 0)),
            float(el.get('width', 0)), float(el.get('height', 0)),
        ))

    grid = [[0] * 6 for _ in range(6)]
    for r in range(6):
        for c in range(6):
            cx, cy = c + 0.5, r + 0.5
            for (x, y, w, h) in white:
                if x <= cx <= x + w and y <= cy <= y + h:
                    grid[r][c] = 1
                    break
    return grid


def compute_policy_path(pol, start, goal, max_len=200):
    """Follow `pol` from `start` to `goal`, returning the ordered cells."""
    path = [start]
    seen = {start}
    pos = start
    while pos != goal and len(path) < max_len:
        action = int(pol[pos])
        if action not in ACTION_DELTA:
            break
        dr, dc = ACTION_DELTA[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        if nxt in seen:
            break
        path.append(nxt)
        seen.add(nxt)
        pos = nxt
    return path


# ===================== SIMULATION GEOMETRY =====================

def cell_xy(cell):
    """(row, col) grid cell -> (x, y) plot coordinates."""
    return (cell[1], cell[0])


def action_between(a, b):
    """Policy action that moves from cell a to adjacent cell b."""
    return DELTA_TO_ACTION[(b[0] - a[0], b[1] - a[1])]


def triangle_vertices(x, y, ux, uy, size):
    """Vertices of a triangle centred at (x,y) pointing along (ux,uy)."""
    perp = (-uy, ux)
    tip   = (x + ux * size,        y + uy * size)
    baseL = (x - ux * size * 0.6 + perp[0] * size * 0.75,
             y - uy * size * 0.6 + perp[1] * size * 0.75)
    baseR = (x - ux * size * 0.6 - perp[0] * size * 0.75,
             y - uy * size * 0.6 - perp[1] * size * 0.75)
    return [tip, baseL, baseR]


def build_frames(path, substeps, start_hold, end_hold):
    """Pre-compute one dict per animation frame describing the robot."""
    frames = []
    x0, y0 = cell_xy(path[0])
    h0 = action_between(path[0], path[1]) if len(path) > 1 else RIGHT

    for k in range(start_hold):
        frames.append(dict(x=x0, y=y0, heading=h0, step=0,
                           arrived_cell=(path[0] if k == 0 else None)))

    for m in range(len(path) - 1):
        a, b = path[m], path[m + 1]
        ax_, ay_ = cell_xy(a)
        bx_, by_ = cell_xy(b)
        heading = action_between(a, b)
        for s in range(1, substeps + 1):
            t = s / substeps
            frames.append(dict(
                x=ax_ + (bx_ - ax_) * t,
                y=ay_ + (by_ - ay_) * t,
                heading=heading, step=m + 1,
                arrived_cell=(path[m + 1] if s == substeps else None),
            ))

    last = frames[-1]
    for _ in range(end_hold):
        frames.append(dict(x=last['x'], y=last['y'], heading=last['heading'],
                           step=last['step'], arrived_cell=None))
    return frames


class SimState:
    """Mutable state carried across animation frames."""

    def __init__(self, path):
        self.trail_x = [cell_xy(path[0])[0]]
        self.trail_y = [cell_xy(path[0])[1]]
        self.visited = set()            # cells already arrived at
        self.recognized = set()         # ArUco ids already recognized
        self.flash = {}                 # ArUco id -> frames left flashing
        self.events = []                # event-log lines
        self.cam_marker = None          # ArUco id currently in the camera


# ===================== VISUALIZATION =====================

# robot / camera colours and sizes
ROBOT_BODY   = '#13476b'
ROBOT_TRI    = '#3da5ff'
CAM_CONE     = '#ffd24a'
CAM_HALF_ANG = 30
CAM_RADIUS   = 1.15
FLASH_FRAMES = 14

SUBSTEPS   = 9
START_HOLD = 8
END_HOLD   = 16


def main():
    save = '--save' in sys.argv

    markers, aruco_dict_name = load_svg_markers(SVG_DIR)
    marker_files = {mid: p for mid, p in markers}
    marker_grid = {}
    for mid in MARKER_TO_CELL:
        if mid in marker_files:
            marker_grid[mid] = np.array(decode_svg_grid(marker_files[mid]))
        else:
            marker_grid[mid] = np.full((6, 6), 0.5)   # missing SVG -> grey

    path = compute_policy_path(policy, START, GOAL)
    cell_to_marker = {cell: mid for mid, cell in MARKER_TO_CELL.items()}
    frames = build_frames(path, SUBSTEPS, START_HOLD, END_HOLD)
    n_moves = len(path) - 1

    print(f"[simulation] ArUco dictionary : {aruco_dict_name}")
    print(f"[simulation] policy path      : {path}  ({len(path)} cells)")
    print(f"[simulation] SVG markers      : {len(markers)} loaded from {SVG_DIR}")
    for mid, cell in sorted(MARKER_TO_CELL.items()):
        print(f"[simulation]   ArUco #{mid} -> fixed cell {cell}")

    # ---------- figure layout ----------
    fig = plt.figure(figsize=(13.5, 8.2))
    fig.suptitle("AlphaBot2 — Policy Navigation & ArUco Marker Recognition "
                 "(simulation)", fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(2, 2, width_ratios=[2.05, 1.0],
                          height_ratios=[1.0, 1.0],
                          left=0.05, right=0.975, top=0.90, bottom=0.06,
                          wspace=0.18, hspace=0.22)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_cam = fig.add_subplot(gs[0, 1])
    ax_log = fig.add_subplot(gs[1, 1])

    # ---------- static map ----------
    ax_map.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax_map.set_ylim(GRID_SIZE - 0.5, -0.5)        # invert: row 0 at the top
    ax_map.set_aspect('equal')
    ax_map.set_xticks(range(GRID_SIZE))
    ax_map.set_yticks(range(GRID_SIZE))
    ax_map.set_xlabel('column')
    ax_map.set_ylabel('row')

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = (r, c)
            if cell in OBSTACLES:
                face = '#1c1c1c'
            elif cell == START:
                face = '#fff0a8'
            elif cell == GOAL:
                face = '#aee5a0'
            else:
                face = '#d4ecf7'
            ax_map.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor=face, edgecolor='#9bb3c0',
                                       linewidth=1.0, zorder=0))
            # policy arrow
            act = int(policy[cell])
            if cell not in OBSTACLES and cell != GOAL and act in ACTION_ARROW:
                ax_map.text(c, r, ACTION_ARROW[act], ha='center', va='center',
                            fontsize=15, color='#6c7a86', zorder=2)

    # planned policy route
    rx = [cell_xy(c)[0] for c in path]
    ry = [cell_xy(c)[1] for c in path]
    ax_map.plot(rx, ry, color='#ff8c1a', linewidth=7, alpha=0.30,
                solid_capstyle='round', zorder=1)

    # ArUco SVG markers drawn on their fixed cells
    marker_borders = {}
    for mid, cell in MARKER_TO_CELL.items():
        r, c = cell
        s = 0.40
        ax_map.imshow(marker_grid[mid], cmap='gray', vmin=0, vmax=1,
                      extent=(c - s, c + s, r + s, r - s),
                      origin='upper', interpolation='nearest', zorder=3)
        border = Rectangle((c - s, r - s), 2 * s, 2 * s, facecolor='none',
                           edgecolor='#888888', linewidth=1.5, zorder=4)
        ax_map.add_patch(border)
        marker_borders[mid] = border
        # id label in the cell corner, on a white chip so it stays readable
        ax_map.text(c - 0.45, r - 0.40, f"#{mid}", ha='left', va='center',
                    fontsize=7.5, color='#222222', zorder=8,
                    bbox=dict(facecolor='white', alpha=0.8, pad=1.0,
                              edgecolor='none'))

    # robot artists
    trail, = ax_map.plot([], [], color=ROBOT_TRI, linewidth=2.6,
                         alpha=0.9, zorder=5)
    cam_wedge = Wedge((0, 0), CAM_RADIUS, 0, 0, facecolor=CAM_CONE,
                      alpha=0.33, edgecolor='none', zorder=4)
    ax_map.add_patch(cam_wedge)
    body = Circle((0, 0), 0.30, facecolor=ROBOT_BODY, edgecolor='white',
                  linewidth=1.6, zorder=6)
    ax_map.add_patch(body)
    tri = Polygon(triangle_vertices(0, 0, 1, 0, 0.30), closed=True,
                  facecolor=ROBOT_TRI, edgecolor='white', linewidth=1.0,
                  zorder=7)
    ax_map.add_patch(tri)

    # ---------- camera panel ----------
    blank_cam = np.full((6, 6), 0.78)
    cam_img = ax_cam.imshow(blank_cam, cmap='gray', vmin=0, vmax=1,
                            interpolation='nearest')
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])
    ax_cam.set_title("camera view\n(scanning...)", fontsize=10)

    # ---------- log panel ----------
    ax_log.axis('off')
    ax_log.set_title("event log", fontsize=10)
    log_text = ax_log.text(0.02, 0.97, '', ha='left', va='top',
                           family='monospace', fontsize=8.2,
                           transform=ax_log.transAxes)

    # ---------- mutable animation state ----------
    state = SimState(path)

    def update(fi):
        fs = frames[fi]
        x, y, heading = fs['x'], fs['y'], fs['heading']
        ux, uy = ACTION_VEC[heading]

        # robot body + heading triangle + camera cone
        body.center = (x, y)
        tri.set_xy(triangle_vertices(x, y, ux, uy, 0.30))
        base = degrees(atan2(uy, ux))
        cam_wedge.set_center((x, y))
        cam_wedge.set_theta1(base - CAM_HALF_ANG)
        cam_wedge.set_theta2(base + CAM_HALF_ANG)

        # trail
        state.trail_x.append(x)
        state.trail_y.append(y)
        trail.set_data(state.trail_x, state.trail_y)

        # arrival -> marker recognition (the localization step)
        cell = fs['arrived_cell']
        if cell is not None and cell not in state.visited:
            state.visited.add(cell)
            r, c = cell
            if cell == START:
                state.events.append(f"start : AlphaBot2 placed at ({r},{c})")
            else:
                state.events.append(f"step {fs['step']:2d}: moved to cell ({r},{c})")
            mid = cell_to_marker.get(cell)
            if mid is not None:
                state.recognized.add(mid)
                state.flash[mid] = FLASH_FRAMES
                state.cam_marker = mid
                tag = 'GOAL reached' if cell == GOAL else 'position confirmed'
                state.events.append(
                    f"   [CAM] recognized ArUco #{mid}  ->  {tag}")

        # marker border styling: flashing / recognized / pending
        for mid, border in marker_borders.items():
            if state.flash.get(mid, 0) > 0:
                state.flash[mid] -= 1
                border.set_edgecolor('#ff2d2d')
                border.set_linewidth(4.0)
            elif mid in state.recognized:
                border.set_edgecolor('#19c419')
                border.set_linewidth(3.0)
            else:
                border.set_edgecolor('#888888')
                border.set_linewidth(1.5)

        # camera panel
        if state.cam_marker is not None:
            m = state.cam_marker
            cam_img.set_data(marker_grid[m])
            ax_cam.set_title(
                f"camera view  —  ArUco #{m}\n"
                f"linked to policy cell {MARKER_TO_CELL[m]}", fontsize=10)
        else:
            cam_img.set_data(blank_cam)
            ax_cam.set_title("camera view\n(scanning...)", fontsize=10)

        # log panel
        log_text.set_text('\n'.join(state.events[-15:]))

        # map title
        step = min(fs['step'], n_moves)
        ax_map.set_title(
            f"step {step}/{n_moves}    heading {ACTION_NAME[heading]}    "
            f"markers recognized {len(state.recognized)}/{len(MARKER_TO_CELL)}",
            fontsize=11)
        return ()

    update(0)
    ani = FuncAnimation(fig, update, frames=len(frames), interval=55,
                        blit=False, repeat=False)

    if save:
        gif = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'simulation_run.gif')
        png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'simulation_run.png')
        ani.save(gif, writer=PillowWriter(fps=18))
        update(len(frames) - 1)
        fig.savefig(png, dpi=110)
        print(f"[simulation] saved {gif}")
        print(f"[simulation] saved {png}")

    plt.show()
    return ani


if __name__ == '__main__':
    main()
