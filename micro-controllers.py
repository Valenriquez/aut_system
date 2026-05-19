import numpy as np
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Load your policy 
with open("policy.json") as f:
    data = json.load(f)

POLICY    = data["policy"]
GOAL      = tuple(data["goal"])
GRID_SIZE = data["grid_size"]

GRID = [
    [0,  0,  0, 'G'],
    [0,  1,  0,  0 ],
    [0,  1,  0,  0 ],
    [0,  0,  0,  0 ],
]

ACTIONS = {
    'up':    (-1,  0),
    'down':  ( 1,  0),
    'left':  ( 0, -1),
    'right': ( 0,  1),
}

NOISE = 0.1

# ── Simulate one episode ──────────────────────────────────────────
def simulate(start=(3,0), max_steps=50):
    pos = list(start)
    path = [tuple(pos)]

    for step in range(max_steps):
        r, c = pos

        if (r, c) == GOAL:
            print(f"GOAL reached in {step} steps!")
            break

        # Look up policy
        action = POLICY.get(f"{r},{c}")
        if not action or action == 'goal':
            break

        # Apply transition noise
        roll = random.random()
        dr, dc = ACTIONS[action]
        if roll < NOISE:
            # drift left
            dr, dc = dc, -dr
        elif roll < 2 * NOISE:
            # drift right
            dr, dc = -dc, dr

        # Move
        nr, nc = r + dr, c + dc

        # Bounce off walls
        rows, cols = GRID_SIZE
        if 0 <= nr < rows and 0 <= nc < cols and GRID[nr][nc] != 1:
            pos = [nr, nc]
        # else stay in place

        path.append(tuple(pos))

    return path

# ── Visualize ─────────────────────────────────────────────────────
def visualize(path):
    rows, cols = GRID_SIZE
    fig, ax = plt.subplots(figsize=(6, 6))

    for r in range(rows):
        for c in range(cols):
            cell = GRID[r][c]
            if cell == 1:
                color = 'black'
            elif (r, c) == GOAL:
                color = 'lightgreen'
            elif (r, c) == tuple(path[0]):
                color = 'lightyellow'
            else:
                color = 'lightblue'

            ax.add_patch(patches.Rectangle(
                (c, rows - r - 1), 1, 1,
                linewidth=1, edgecolor='gray', facecolor=color
            ))

            # Draw policy arrows
            action = POLICY.get(f"{r},{c}")
            if action and action not in ['goal', None] and cell != 1:
                arrows = {'up': '↑', 'down': '↓',
                          'left': '←', 'right': '→'}
                ax.text(c + 0.5, rows - r - 0.5,
                        arrows[action],
                        ha='center', va='center', fontsize=14)

    # Draw path
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        ax.annotate("",
            xy=(c2 + 0.5, rows - r2 - 0.5),
            xytext=(c1 + 0.5, rows - r1 - 0.5),
            arrowprops=dict(arrowstyle="->", color="red", lw=2)
        )

    # Mark start and goal
    ax.text(path[0][1] + 0.5, rows - path[0][0] - 0.5,
            'S', ha='center', va='center',
            fontsize=14, color='orange', fontweight='bold')
    ax.text(GOAL[1] + 0.5, rows - GOAL[0] - 0.5,
            'G', ha='center', va='center',
            fontsize=14, color='green', fontweight='bold')

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title(f"Simulation — {len(path)-1} steps")
    plt.tight_layout()
    plt.savefig("simulation.png")
    plt.show()
    print("Saved to simulation.png")

# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = simulate(start=(3, 0))
    print(f"Path taken: {path}")
    visualize(path)