#!/usr/bin/env python3
"""
Q-learning trainer for the AlphaBot2 grid maze.

This is the OFFLINE part of the pipeline. It is plain numpy — no ROS.
It learns a tabular Q-function on a simulated 7x7 grid MDP, extracts the
greedy policy, validates that the policy actually reaches the goal, and
saves it to `learned_policy.npy` for the ROS 2 runner to load.

Why offline?  Tabular Q-learning needs thousands of episodes and crashes
into walls during exploration. You do that in a simulator (here), not on
a physical robot. The robot only ever executes the converged greedy policy.

Difference from value iteration (your old code):
  * Value iteration KNOWS the model (transitions + rewards) and solves the
    Bellman optimality equation directly.
  * Q-learning does NOT know the model. It only observes (s, a, r, s')
    transitions and bootstraps:
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
    The obstacle map below is used ONLY to simulate the environment's
    responses — the learner never reads it directly.
"""

import numpy as np

# ===================== GRID / WORLD (same as robot) =====================
GRID_SIZE = 7
START     = (0, 0)
GOAL      = (6, 6)
OBSTACLES = {
    (1, 0), (1, 2), (1, 3), (1, 4), (1, 6),
    (3, 1), (3, 2), (3, 3), (3, 5),
    (4, 3), (4, 5),
    (5, 1), (5, 5),
    (6, 1), (6, 3), (6, 5),
}

# Action encoding — IDENTICAL to policy_runner so the policy array is drop-in.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS      = [UP, DOWN, LEFT, RIGHT]
ACTION_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
ARROW        = {UP: '^', DOWN: 'v', LEFT: '<', RIGHT: '>', -1: ' '}

# ===================== REWARD DESIGN =====================
GOAL_REWARD  = 100.0   # terminal: arriving at the goal cell
STEP_PENALTY = -1.0    # every legal move → pushes toward the SHORTEST path
WALL_PENALTY = -10.0   # hit a wall/obstacle → bounce back, stay in place

# ===================== HYPERPARAMETERS =====================
ALPHA          = 0.10   # learning rate
GAMMA          = 0.95   # discount factor
N_EPISODES     = 3000
MAX_STEPS      = 200    # episode horizon (prevents wandering forever)
EPS_START      = 1.00   # epsilon-greedy: start fully exploratory
EPS_END        = 0.05
EPS_DECAY_EPS  = 2000   # episodes over which epsilon decays linearly
SEED           = 0


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT  (the simulator — Q-learning treats this as a black box)
# ─────────────────────────────────────────────────────────────
def step(state, action):
    """Return (next_state, reward, done). Deterministic transitions."""
    dr, dc = ACTION_DELTA[action]
    nr, nc = state[0] + dr, state[1] + dc
    nxt = (nr, nc)

    # off-grid or into an obstacle → bounce back, penalty
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) or nxt in OBSTACLES:
        return state, WALL_PENALTY, False

    if nxt == GOAL:
        return nxt, GOAL_REWARD, True

    return nxt, STEP_PENALTY, False


# ─────────────────────────────────────────────────────────────
# Q-LEARNING
# ─────────────────────────────────────────────────────────────
def train():
    rng = np.random.default_rng(SEED)
    Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)), dtype=float)

    returns = []
    for ep in range(N_EPISODES):
        # linear epsilon decay
        frac    = min(1.0, ep / EPS_DECAY_EPS)
        epsilon = EPS_START + frac * (EPS_END - EPS_START)

        state = START
        total = 0.0
        for _ in range(MAX_STEPS):
            # epsilon-greedy action selection
            if rng.random() < epsilon:
                action = int(rng.integers(len(ACTIONS)))
            else:
                action = int(np.argmax(Q[state[0], state[1]]))

            nxt, reward, done = step(state, action)
            total += reward

            # Q-learning (off-policy TD) update
            best_next = 0.0 if done else float(np.max(Q[nxt[0], nxt[1]]))
            td_target = reward + GAMMA * best_next
            td_error  = td_target - Q[state[0], state[1], action]
            Q[state[0], state[1], action] += ALPHA * td_error

            state = nxt
            if done:
                break
        returns.append(total)

    return Q, returns


# ─────────────────────────────────────────────────────────────
# POLICY EXTRACTION + VALIDATION
# ─────────────────────────────────────────────────────────────
def extract_policy(Q):
    """Greedy policy from Q. Obstacles and goal → -1 (matches old format)."""
    policy = -np.ones((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r, c) in OBSTACLES or (r, c) == GOAL:
                continue
            policy[r, c] = int(np.argmax(Q[r, c]))
    return policy


def rollout(policy, max_len=200):
    """Follow the greedy policy from START; return the visited path."""
    path, seen, pos = [START], {START}, START
    while pos != GOAL and len(path) < max_len:
        a = int(policy[pos])
        if a not in ACTION_DELTA:
            break
        dr, dc = ACTION_DELTA[a]
        nxt = (pos[0] + dr, pos[1] + dc)
        if (not (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE)
                or nxt in OBSTACLES or nxt in seen):
            break
        path.append(nxt); seen.add(nxt); pos = nxt
    return path


def pretty_grid(policy, path):
    """ASCII map: S goal G, # obstacle, * path, arrow elsewhere."""
    path_set = set(path)
    lines = []
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            if (r, c) == START:     row.append('S')
            elif (r, c) == GOAL:    row.append('G')
            elif (r, c) in OBSTACLES: row.append('#')
            elif (r, c) in path_set:  row.append(ARROW[int(policy[r, c])])
            else:                     row.append('.')
        lines.append(' '.join(row))
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────
def main():
    print('Training Q-learning ...')
    Q, returns = train()
    policy = extract_policy(Q)
    path   = rollout(policy)

    reached = (path[-1] == GOAL)
    print(f'\nLast-100-episode mean return: {np.mean(returns[-100:]):.1f}')
    print(f'Greedy path length: {len(path)} cells   reaches goal: {reached}')
    print(f'Path: {path}\n')
    print(pretty_grid(policy, path))

    print('\nLearned policy array (paste-ready):')
    print('policy = np.array([')
    for r in range(GRID_SIZE):
        print('    [' + ', '.join(f'{int(policy[r, c]):2d}' for c in range(GRID_SIZE)) + '],')
    print('], dtype=int)')

    if reached:
        np.save('learned_policy.npy', policy)
        print('\nSaved → learned_policy.npy')
    else:
        print('\n!! Policy did not reach goal — increase N_EPISODES or check rewards.')


if __name__ == '__main__':
    main()