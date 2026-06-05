import numpy as np
import json

# ── 1. DEFINE YOUR GRID ──────────────────────────────────────────────────────
# 0 = free cell, 1 = wall, 'G' = goal
# 7x7 maze (same layout as the Q-learning files): start (0,0), goal (6,6).
# Walls correspond to the OBSTACLES set; 'G' marks the goal cell.
GRID = [
    [0,  0,  0,  0,  0,  0,  0 ],
    [1,  0,  1,  1,  1,  0,  1 ],
    [0,  0,  0,  0,  0,  0,  0 ],
    [0,  1,  1,  1,  0,  1,  0 ],
    [0,  0,  0,  1,  0,  1,  0 ],
    [0,  1,  0,  0,  0,  1,  0 ],
    [0,  1,  0,  1,  0,  1, 'G'],
]
ROWS = len(GRID)
COLS = len(GRID[0])

# Find goal position
GOAL = next((r, c) for r in range(ROWS)
                    for c in range(COLS) if GRID[r][c] == 'G')

# ── 2. DEFINE MDP PARAMETERS ─────────────────────────────────────────────────
ACTIONS = {
    'up':    (-1,  0),
    'down':  ( 1,  0),
    'left':  ( 0, -1),
    'right': ( 0,  1),
}

GAMMA   = 0.9    # discount factor
NOISE   = 0.1    # 10% chance of drifting sideways (from your slides)
REWARD_GOAL = 100
REWARD_STEP = -20
REWARD_WALL = -10
THRESHOLD   = 0.001  # stop when values change less than this

# ── 3. HELPER FUNCTIONS ───────────────────────────────────────────────────────
def is_free(r, c):
    """True if cell exists and is not a wall."""
    return 0 <= r < ROWS and 0 <= c < COLS and GRID[r][c] != 1

def get_reward(r, c):
    if (r, c) == GOAL:
        return REWARD_GOAL
    elif not is_free(r, c):
        return REWARD_WALL
    return REWARD_STEP

def transition(r, c, action):
    """
    Returns list of (probability, next_r, next_c).
    0.8 → intended direction
    0.1 → drift left of intended
    0.1 → drift right of intended
    If the robot hits a wall, it stays in place.
    """
    dr, dc = ACTIONS[action]
    # Define the three possible outcomes
    intended   = (dr, dc)
    drift_left  = ( dc, -dr)   # rotate 90° left
    drift_right = (-dc,  dr)   # rotate 90° right

    outcomes = [
        (1 - 2 * NOISE, intended),
        (NOISE,          drift_left),
        (NOISE,          drift_right),
    ]

    results = []
    for prob, (mr, mc) in outcomes:
        nr, nc = r + mr, c + mc
        if not is_free(nr, nc):
            nr, nc = r, c      # bounce back to current cell
        results.append((prob, nr, nc))
    return results

# ── 4. VALUE ITERATION ────────────────────────────────────────────────────────
def value_iteration():
    # Initialize V(s) = 0 for all states
    V = np.zeros((ROWS, COLS))

    iteration = 0
    while True:
        delta = 0
        new_V = V.copy()

        for r in range(ROWS):
            for c in range(COLS):
                # Skip walls and goal (goal value is fixed)
                if GRID[r][c] == 1:
                    continue
                if (r, c) == GOAL:
                    new_V[r][c] = REWARD_GOAL
                    continue

                # Bellman update: try every action, keep the best
                action_values = []
                for action in ACTIONS:
                    total = 0
                    for prob, nr, nc in transition(r, c, action):
                        total += prob * (get_reward(nr, nc) + GAMMA * V[nr][nc])
                    action_values.append(total)

                new_V[r][c] = max(action_values)
                delta = max(delta, abs(new_V[r][c] - V[r][c]))

        V = new_V
        iteration += 1

        if delta < THRESHOLD:
            print(f"Converged after {iteration} iterations")
            break

    return V

# ── 5. POLICY EXTRACTION ──────────────────────────────────────────────────────
def extract_policy(V):
    """
    For each cell, pick the action that gives the highest expected value.
    This is the argmax from your slides.
    """
    policy = {}

    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r][c] == 1:
                policy[f"{r},{c}"] = None   # wall
                continue
            if (r, c) == GOAL:
                policy[f"{r},{c}"] = 'goal'
                continue

            best_action = None
            best_value  = float('-inf')

            for action in ACTIONS:
                total = 0
                for prob, nr, nc in transition(r, c, action):
                    total += prob * (get_reward(nr, nc) + GAMMA * V[nr][nc])
                if total > best_value:
                    best_value  = total
                    best_action = action

            policy[f"{r},{c}"] = best_action

    return policy

# ── 6. SAVE TO FILE ───────────────────────────────────────────────────────────
def save_policy(policy, V, path="policy.json"):
    output = {
        "policy": policy,
        "values": {f"{r},{c}": round(V[r][c], 2)
                   for r in range(ROWS) for c in range(COLS)},
        "grid_size": [ROWS, COLS],
        "goal": list(GOAL),
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Policy saved to {path}")

# ── 7. RUN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running value iteration...")
    V = value_iteration()

    print("\nValue grid:")
    print(np.round(V, 1))

    policy = extract_policy(V)

    print("\nPolicy grid:")
    for r in range(ROWS):
        row = [policy[f"{r},{c}"][0].upper()
               if policy[f"{r},{c}"] and policy[f"{r},{c}"] != 'goal'
               else ('G' if policy[f"{r},{c}"] == 'goal' else 'X')
               for c in range(COLS)]
        print(row)

    save_policy(policy, V)