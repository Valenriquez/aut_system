# Results

All learning results are reproducible (numpy, seed = 0). The maze is a 7×7 grid,
start (0,0), goal (6,6), with 16 obstacle cells. The provably shortest route is
**13 cells**; "convergence episode" = the first training episode after which the
greedy policy reaches the goal along that 13-cell optimal path.

## 1. Reward-shaping ablation

We toggled the two reward-shaping components — potential-based goal shaping
(BFS distance) and the 8-sector danger penalty — and measured how fast learning
converged and whether the final route stayed optimal.

| Goal shaping | Danger penalty | Convergence (episode) | Path length | Reaches goal |
|:---:|:---:|:---:|:---:|:---:|
| on  | on  | 5   | 13 | yes |
| on  | off | **2**  | 13 | yes |
| off | on  | 63  | 13 | yes |
| off | off | 46  | 13 | yes |

**Findings.**
- **Goal shaping is the real accelerant:** it cuts convergence from ~46 episodes
  to 2–5 — roughly a **10–20× speed-up** — while leaving the final 13-cell route
  unchanged. This is the expected behaviour of potential-based shaping: it
  provably cannot change the optimal policy, only how fast it is found.
- **The danger penalty alone slightly *hurts* convergence** (63 vs 46 episodes,
  and 5 vs 2 when combined with shaping). It biases exploration away from
  boxed-in cells, which on this small, fully-traversable maze is an unnecessary
  drag rather than a help. It is retained as a descriptor/safety bias, not a
  speed feature.
- **Every configuration converges to the same optimal 13-cell path.** Across all
  four runs the learned policy differed on at most one *off-path* cell, never on
  a cell the robot actually visits.

## 2. Hyperparameter sensitivity

With shaping enabled, we swept the learning rate (α), discount (γ) and the
ε-greedy decay horizon. All runs reached the optimal 13-cell path.

| α | conv. ep | | γ | conv. ep | | ε-decay | conv. ep |
|:---:|:---:|---|:---:|:---:|---|:---:|:---:|
| 0.02 | 5 | | 0.80 | 5 | | 500  | 8 |
| 0.05 | 5 | | 0.90 | 5 | | 1000 | 7 |
| 0.10 | 5 | | 0.95 | 5 | | 2000 | 5 |
| 0.30 | 5 | | 0.99 | 5 | | 3000 | 5 |
| 0.60 | 5 | |      |   | |      |   |

**Finding.** Convergence is **insensitive to α and γ** (constant at 5 episodes
across two orders of magnitude of α) and only mildly sensitive to the
exploration schedule (faster ε-decay → slightly slower first solve, 8 vs 5).
The dense shaping signal dominates the value estimates, so the algorithm is
robust to hyperparameter choice on this task — a practical advantage for
reproducibility.

> Note: the file ships with `N_EPISODES = 3000`, but the policy is stable from
> ~episode 5. ~99% of configured training is redundant on this maze; 100–200
> episodes would yield an identical policy.

## 3. Learned policy and path

Arrow = greedy action, `S` start, `G` goal, `#` obstacle, `.` free cell:

```
S v . . . . .
# v # # # . #
. > > > > > v
. # # # . # v
. . . # . # v
. # . . . # v
. # . # . # G
```

Executed path (13 cells, optimal):
`(0,0)(0,1)(1,1)(2,1)(2,2)(2,3)(2,4)(2,5)(2,6)(3,6)(4,6)(5,6)(6,6)`

## 4. 8-sector obstacle descriptor (danger map)

`danger_level(cell)` = number of the 8 surrounding sectors that are blocked
(obstacle or off-grid), 0–8. Higher = more boxed-in:

```
6 5 5 6 5 5 6
# 2 # # # 2 #
5 4 5 5 4 3 5
4 # # # 4 # 5
5 3 5 # 5 # 6
5 # 4 2 5 # 6
7 # 6 # 6 # G
```

Corners and obstacle-adjacent cells score highest (e.g. (6,0) = 7). This is the
compact "object distribution" descriptor [5] that feeds the danger penalty.

## 5. Limitations

- **No odometry / no localisation.** The robot exposes only the 5 IR line
  sensors. At runtime it tracks its grid cell by **dead reckoning** — it counts
  the cells it *believes* it has crossed and trusts the timing constants
  (`CELL_SIZE / speed × FWD_TIME_SCALE`). There is no feedback that confirms
  "I am at cell (r,c)". A single mis-timed move or an over/under-rotated turn
  makes the tracked position wrong, and the robot then looks up the policy for
  the wrong cell. This is the dominant source of runtime failure.

- **Timing-based motion is open-loop.** Because commanded speed ≠ actual ground
  speed, forward and turn distances depend on hand-tuned fudge factors
  (`FWD_TIME_SCALE = 1.5`, `TURN_TIME_SCALE = 1.3`). These must be re-calibrated
  for any change in surface, battery level, or the simulator's `CELL_SIZE`.

- **Line sensors steer but do not localise.** They keep the robot centred on the
  tape (fuzzy steering [4]) and detect line-loss for recovery [3], but they
  cannot identify *which* intersection the robot is at, so they cannot correct
  dead-reckoning drift.

- **The BFS shaping oracle assumes a fully known map.** The potential function
  is built from a BFS over the known obstacle layout. This accelerates training
  but is only available because the map is given in advance; on an unknown map
  the shaping term would have to be replaced by a sensor-derived heuristic
  (e.g. Manhattan distance).

- **No reactive obstacle avoidance / wall following.** True U-trap escape and
  wall following need proximity/range sensors this robot does not publish; only
  the planning-level guarantee (the policy never selects a move into a known
  obstacle) and the line-loss watchdog are implemented.

### Suggested fix for the localisation limitation
Detecting each fully-black intersection ("all 5 sensors on tape") would let the
runner **re-synchronise its cell estimate at every crossing** instead of trusting
the timer — closing most of the dead-reckoning gap using only the existing
sensors. Fusing the simulator's `/odom` topic (if available) would close the rest.
