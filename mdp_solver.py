import numpy as np
import json

# 0 = free cell, 1 = wall, 'G' = goal
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

GOAL = next((r, c) for r in range(ROWS)
                    for c in range(COLS) if GRID[r][c] == 'G')

ACTIONS = {
    'up':    (-1,  0),
    'down':  ( 1,  0),
    'left':  ( 0, -1),
    'right': ( 0,  1),
}

GAMMA   = 0.9
NOISE   = 0.1
REWARD_GOAL = 100
REWARD_STEP = -1
REWARD_WALL = -100
THRESHOLD   = 0.001


def is_free(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and GRID[r][c] != 1

def get_reward(r, c):
    return REWARD_GOAL if (r, c) == GOAL else REWARD_STEP

def transition(r, c, action):
    dr, dc = ACTIONS[action]
    intended    = (dr, dc)
    drift_left  = ( dc, -dr)
    drift_right = (-dc,  dr)

    outcomes = [
        (1 - 2 * NOISE, intended),
        (NOISE,          drift_left),
        (NOISE,          drift_right),
    ]

    results = []
    for prob, (mr, mc) in outcomes:
        nr, nc = r + mr, c + mc
        if not is_free(nr, nc):
            results.append((prob, REWARD_WALL, r, c))
        else:
            results.append((prob, get_reward(nr, nc), nr, nc))
    return results

def value_iteration():
    V = np.zeros((ROWS, COLS))

    iteration = 0
    while True:
        delta = 0
        new_V = V.copy()

        for r in range(ROWS):
            for c in range(COLS):
                if GRID[r][c] == 1:
                    continue
                if (r, c) == GOAL:
                    new_V[r][c] = REWARD_GOAL
                    continue

                action_values = []
                for action in ACTIONS:
                    total = 0
                    for prob, reward, nr, nc in transition(r, c, action):
                        total += prob * (reward + GAMMA * V[nr][nc])
                    action_values.append(total)

                new_V[r][c] = max(action_values)
                delta = max(delta, abs(new_V[r][c] - V[r][c]))

        V = new_V
        iteration += 1

        if delta < THRESHOLD:
            print(f"Converged after {iteration} iterations")
            break

    return V

def extract_policy(V):
    policy = {}

    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r][c] == 1:
                policy[f"{r},{c}"] = None
                continue
            if (r, c) == GOAL:
                policy[f"{r},{c}"] = 'goal'
                continue

            best_action = None
            best_value  = float('-inf')

            for action in ACTIONS:
                total = 0
                for prob, reward, nr, nc in transition(r, c, action):
                    total += prob * (reward + GAMMA * V[nr][nc])
                if total > best_value:
                    best_value  = total
                    best_action = action

            policy[f"{r},{c}"] = best_action

    return policy

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
