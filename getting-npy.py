#!/usr/bin/env python3
"""
Produces policy.npy in the SAME FOLDER as this script.
Run with:  python3 final.py
"""

import os
import numpy as np

# ============================================
# ENVIRONMENT
# ============================================
GRID_SIZE = 7
GAMMA = 0.9
THETA = 0.001

STEP_REWARD = -1
GOAL_REWARD = 100

P_INTENDED = 0.8
P_DRIFT = 0.1

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = {
    UP:    (-1, 0),
    DOWN:  (1, 0),
    LEFT:  (0, -1),
    RIGHT: (0, 1),
}
PERPENDICULAR = {
    UP:    (LEFT, RIGHT),
    DOWN:  (LEFT, RIGHT),
    LEFT:  (UP, DOWN),
    RIGHT: (UP, DOWN),
}
ACTION_SYMBOLS = {UP: '^', DOWN: 'v', LEFT: '<', RIGHT: '>'}

goal_state = (6, 6)
obstacles = {
    (1, 0), (1, 2), (1, 3), (1, 4), (1, 6),
    (3, 1), (3, 2), (3, 3), (3, 5),
    (4, 3), (4, 5),
    (5, 1), (5, 5),
    (6, 1), (6, 3), (6, 5),
}

# ============================================
# HELPERS
# ============================================
def is_valid_state(state):
    r, c = state
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and state not in obstacles

def move(state, action):
    dr, dc = ACTIONS[action]
    nxt = (state[0] + dr, state[1] + dc)
    return nxt if is_valid_state(nxt) else state

def get_reward(nxt):
    return GOAL_REWARD if nxt == goal_state else STEP_REWARD

def get_transitions(state, action):
    out = []
    ni = move(state, action)
    out.append((P_INTENDED, ni, get_reward(ni)))
    da, db = PERPENDICULAR[action]
    na = move(state, da); out.append((P_DRIFT, na, get_reward(na)))
    nb = move(state, db); out.append((P_DRIFT, nb, get_reward(nb)))
    return out

def expected_action_value(state, action, V):
    return sum(p * (r + GAMMA * V[ns]) for p, ns, r in get_transitions(state, action))

# ============================================
# VALUE ITERATION
# ============================================
V = np.zeros((GRID_SIZE, GRID_SIZE))
policy = np.full((GRID_SIZE, GRID_SIZE), -1)

iteration = 0
while True:
    delta = 0
    new_V = np.copy(V)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            s = (row, col)
            if s in obstacles or s == goal_state:
                continue
            best = max(expected_action_value(s, a, V) for a in ACTIONS)
            new_V[s] = best
            delta = max(delta, abs(V[s] - best))
    V = new_V
    iteration += 1
    if delta < THETA:
        break

print(f"Converged after {iteration} iterations")

# ============================================
# POLICY EXTRACTION
# ============================================
for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        s = (row, col)
        if s in obstacles or s == goal_state:
            continue
        best_a, best_v = None, -float('inf')
        for a in ACTIONS:
            v = expected_action_value(s, a, V)
            if v > best_v:
                best_v, best_a = v, a
        policy[s] = best_a

print("\nOptimal Policy:\n")
for row in range(GRID_SIZE):
    line = ""
    for col in range(GRID_SIZE):
        s = (row, col)
        if s in obstacles:
            line += " X "
        elif s == goal_state:
            line += " G "
        else:
            line += f" {ACTION_SYMBOLS[policy[s]]} "
    print(line)

# ============================================
# SAVE policy.npy NEXT TO THIS SCRIPT
# ============================================
here = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(here, 'policy.npy')
np.save(out_path, policy)

print("\n" + "=" * 60)
print(f"  policy.npy saved to:")
print(f"  {out_path}")
print("=" * 60)