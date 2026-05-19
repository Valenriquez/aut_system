import numpy as np
import matplotlib.pyplot as plt

# ============================================
# ENVIRONMENT SETTINGS
# ============================================

GRID_SIZE = 6
GAMMA = 0.9
THETA = 0.001

# Rewards
STEP_REWARD = -1
GOAL_REWARD = 100
OBSTACLE_REWARD = -100

# ============================================
# ACTIONS
# ============================================

ACTIONS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1)    # RIGHT
}

ACTION_SYMBOLS = {
    0: '^',
    1: 'v',
    2: '<',
    3: '>'
}

# ============================================
# WORLD DEFINITION
# ============================================

goal_state = (5, 5)

obstacles = {
    (1, 1),
    (2, 1),
    (3, 1),
    (3, 2),
    (3, 3)
}

# ============================================
# INITIALIZE VALUE FUNCTION
# ============================================

V = np.zeros((GRID_SIZE, GRID_SIZE))
policy = np.full((GRID_SIZE, GRID_SIZE), -1)

# ============================================
# HELPER FUNCTIONS
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



def get_next_state(state, action):
    dr, dc = ACTIONS[action]

    next_state = (state[0] + dr, state[1] + dc)

    if not is_valid_state(next_state):
        return state

    return next_state



def get_reward(state, next_state):
    if next_state == goal_state:
        return GOAL_REWARD

    if next_state in obstacles:
        return OBSTACLE_REWARD

    return STEP_REWARD

# ============================================
# VALUE ITERATION
# ============================================

iteration = 0

while True:
    delta = 0

    new_V = np.copy(V)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            state = (row, col)

            if state in obstacles:
                continue

            if state == goal_state:
                continue

            action_values = []

            for action in ACTIONS:
                next_state = get_next_state(state, action)

                reward = get_reward(state, next_state)

                value = reward + GAMMA * V[next_state]

                action_values.append(value)

            best_action_value = max(action_values)

            new_V[state] = best_action_value

            delta = max(delta, abs(V[state] - new_V[state]))

    V = new_V

    iteration += 1

    if delta < THETA:
        break

print(f"Converged after {iteration} iterations")

# ============================================
# EXTRACT POLICY
# ============================================

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):

        state = (row, col)

        if state in obstacles:
            continue

        if state == goal_state:
            continue

        best_action = None
        best_value = -float('inf')

        for action in ACTIONS:
            next_state = get_next_state(state, action)

            reward = get_reward(state, next_state)

            value = reward + GAMMA * V[next_state]

            if value > best_value:
                best_value = value
                best_action = action

        policy[state] = best_action

# ============================================
# PRINT RESULTS
# ============================================

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
            action = policy[state]
            line += f" {ACTION_SYMBOLS[action]} "

    print(line)

# ============================================
# VISUALIZATION
# ============================================

fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(V, cmap='viridis')

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):

        state = (row, col)

        if state in obstacles:
            ax.text(col, row, 'X', ha='center', va='center', color='red', fontsize=20)

        elif state == goal_state:
            ax.text(col, row, 'G', ha='center', va='center', color='white', fontsize=20)

        else:
            action = policy[state]
            ax.text(
                col,
                row,
                ACTION_SYMBOLS[action],
                ha='center',
                va='center',
                color='white',
                fontsize=16
            )

plt.title("Robot Decision Policy using Value Iteration")
plt.colorbar()
plt.show() 