import numpy as np
import matplotlib.pyplot as plt

# ============================================
# ENVIRONMENT SETTINGS
# ============================================

GRID_SIZE = 7
GAMMA = 0.9
THETA = 0.001

# Rewards (as per spec)
STEP_REWARD = -1
GOAL_REWARD = 100

# Stochastic transition probabilities (as per spec)
P_INTENDED = 0.8
P_DRIFT = 0.1  # to each perpendicular side

# ============================================
# ACTIONS
# ============================================

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

ACTIONS = {
    UP:    (-1, 0),
    DOWN:  (1, 0),
    LEFT:  (0, -1),
    RIGHT: (0, 1),
}

ACTION_SYMBOLS = {UP: '^', DOWN: 'v', LEFT: '<', RIGHT: '>'}

# For each action, the two perpendicular ("drift") actions
PERPENDICULAR = {
    UP:    (LEFT, RIGHT),
    DOWN:  (LEFT, RIGHT),
    LEFT:  (UP, DOWN),
    RIGHT: (UP, DOWN),
}

# ============================================
# WORLD DEFINITION
# ============================================

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
    if r < 0 or r >= GRID_SIZE:
        return False
    if c < 0 or c >= GRID_SIZE:
        return False
    if state in obstacles:
        return False
    return True


def move(state, action):
    """Deterministic move in `action` direction. Bounces back if blocked."""
    dr, dc = ACTIONS[action]
    next_state = (state[0] + dr, state[1] + dc)
    if not is_valid_state(next_state):
        return state
    return next_state


def get_reward(next_state):
    if next_state == goal_state:
        return GOAL_REWARD
    return STEP_REWARD


def get_transitions(state, action):
    """
    Returns the stochastic transitions for taking `action` from `state`.
    Each element is (probability, next_state, reward).

      - 0.8: intended direction
      - 0.1: drift to one perpendicular side
      - 0.1: drift to the other perpendicular side
    """
    transitions = []

    ns_intended = move(state, action)
    transitions.append((P_INTENDED, ns_intended, get_reward(ns_intended)))

    drift_a, drift_b = PERPENDICULAR[action]
    ns_a = move(state, drift_a)
    transitions.append((P_DRIFT, ns_a, get_reward(ns_a)))
    ns_b = move(state, drift_b)
    transitions.append((P_DRIFT, ns_b, get_reward(ns_b)))

    return transitions


def expected_action_value(state, action, V):
    """Expected value E[r + gamma * V(s')] under stochastic transitions."""
    total = 0.0
    for prob, next_state, reward in get_transitions(state, action):
        total += prob * (reward + GAMMA * V[next_state])
    return total


# ============================================
# VALUE ITERATION (stochastic Bellman update)
# ============================================

V = np.zeros((GRID_SIZE, GRID_SIZE))
policy = np.full((GRID_SIZE, GRID_SIZE), -1)

iteration = 0
while True:
    delta = 0
    new_V = np.copy(V)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            state = (row, col)
            if state in obstacles or state == goal_state:
                continue

            action_values = [expected_action_value(state, a, V) for a in ACTIONS]
            best = max(action_values)
            new_V[state] = best
            delta = max(delta, abs(V[state] - best))

    V = new_V
    iteration += 1
    if delta < THETA:
        break

print(f"Converged after {iteration} iterations")

# ============================================
# EXTRACT POLICY (greedy w.r.t. expected value)
# ============================================

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        state = (row, col)
        if state in obstacles or state == goal_state:
            continue
        best_action, best_value = None, -float('inf')
        for action in ACTIONS:
            v = expected_action_value(state, action, V)
            if v > best_value:
                best_value = v
                best_action = action
        policy[state] = best_action

print("\nValue Function:\n")
print(np.round(V, 1))

print("\nOptimal Policy:\n")
for row in range(GRID_SIZE):
    line = ""
    for col in range(GRID_SIZE):
        state = (row, col)
        if state in obstacles:
            line += " X "
        elif state == goal_state:
            line += " G "
        else:
            line += f" {ACTION_SYMBOLS[policy[state]]} "
    print(line)


# ============================================
# MICROSIMULATION
# ============================================
# Roll out the learned policy in the stochastic environment.
# At each step we sample which direction actually happens.

def step_stochastic(state, action, rng):
    """Sample an actual next state given the intended action."""
    r = rng.random()
    if r < P_INTENDED:
        actual = action
    elif r < P_INTENDED + P_DRIFT:
        actual = PERPENDICULAR[action][0]
    else:
        actual = PERPENDICULAR[action][1]
    return move(state, actual)


def run_episode(start, policy, rng, max_steps=200):
    """Run one episode. Returns (trajectory, total_reward, reached_goal)."""
    state = start
    trajectory = [state]
    total_reward = 0
    for _ in range(max_steps):
        if state == goal_state:
            break
        action = int(policy[state])
        next_state = step_stochastic(state, action, rng)
        total_reward += get_reward(next_state)
        state = next_state
        trajectory.append(state)
    return trajectory, total_reward, state == goal_state


rng = np.random.default_rng(42)
N_EPISODES = 500
start = (0, 0)

trajectories, rewards, steps_to_goal = [], [], []
successes = 0

for _ in range(N_EPISODES):
    traj, total_r, success = run_episode(start, policy, rng)
    trajectories.append(traj)
    rewards.append(total_r)
    if success:
        successes += 1
        steps_to_goal.append(len(traj) - 1)

print(f"\n--- Microsimulation: {N_EPISODES} episodes from {start} ---")
print(f"Success rate:      {successes / N_EPISODES:.1%}")
print(f"Avg total reward:  {np.mean(rewards):.2f}")
if steps_to_goal:
    print(f"Avg steps to goal: {np.mean(steps_to_goal):.2f}")
    print(f"Min / Max steps:   {min(steps_to_goal)} / {max(steps_to_goal)}")


# ============================================
# VISUALIZATION
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Left panel: value function + policy arrows ---
ax = axes[0]
ax.imshow(V, cmap='viridis')
for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        state = (row, col)
        if state in obstacles:
            ax.text(col, row, 'X', ha='center', va='center', color='red', fontsize=20)
        elif state == goal_state:
            ax.text(col, row, 'G', ha='center', va='center', color='white', fontsize=20)
        else:
            ax.text(col, row, ACTION_SYMBOLS[policy[state]],
                    ha='center', va='center', color='white', fontsize=16)
ax.set_title("Value Function + Optimal Policy\n(stochastic 0.8 / 0.1 / 0.1)")
ax.set_xticks(range(GRID_SIZE))
ax.set_yticks(range(GRID_SIZE))

# --- Right panel: visit heatmap + sample trajectories ---
ax = axes[1]
visit_count = np.zeros((GRID_SIZE, GRID_SIZE))
for traj in trajectories:
    for s in traj:
        visit_count[s] += 1

ax.imshow(visit_count, cmap='hot')

# Overlay 10 trajectories (with small jitter so overlapping lines are visible)
for i, traj in enumerate(trajectories[:10]):
    ys = [s[0] + rng.normal(0, 0.06) for s in traj]
    xs = [s[1] + rng.normal(0, 0.06) for s in traj]
    ax.plot(xs, ys, alpha=0.5, linewidth=1.2)

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        state = (row, col)
        if state in obstacles:
            ax.text(col, row, 'X', ha='center', va='center', color='cyan', fontsize=18)
        elif state == goal_state:
            ax.text(col, row, 'G', ha='center', va='center', color='lime', fontsize=18)

ax.set_title(f"Visit Heatmap + 10 Sample Trajectories\n({N_EPISODES} episodes from {start})")
ax.set_xticks(range(GRID_SIZE))
ax.set_yticks(range(GRID_SIZE))

plt.tight_layout()
plt.savefig('/home/claude/value_iter_stochastic.png', dpi=110, bbox_inches='tight')
plt.show()
np.save('policy.npy', policy)
