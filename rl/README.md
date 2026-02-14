# Reinforcement Learning

Implementations of core reinforcement learning algorithms based on Sutton and Barto's *Reinforcement Learning: An Introduction*.

## Structure

### `dp/` - Dynamic Programming (Chapter 4)
Policy iteration and value iteration for environments with known dynamics.

- **`dp.py`** - `DP` class implementing policy evaluation, policy improvement, and value iteration
- **Examples:**
  - `small_grid_world.py` - 4x4 grid world (Example 4.1)
  - `grid_world.py` - 5x5 grid world with teleporting states (Example 3.5)
  - `car_rental.py` - Jack's car rental (Example 4.2)
  - `gambler.py` - Gambler's problem (Example 4.3)

### `mc/` - Monte Carlo Methods (Chapter 5)
Off-policy Monte Carlo control with weighted importance sampling.

- **`mc.py`** - `MC` class implementing off-policy MC with discounting-aware option
- **Examples:**
  - `race_track.py` - Race track problem (Exercise 5.12)

### `td/` - Temporal-Difference Learning (Chapters 6-8)
Tabular and function approximation TD methods.

- **`td.py`** - `TD` class implementing Sarsa, Q-learning, Expected Sarsa, and n-step variants
- **`td_func.py`** - `TD` with linear function approximation (Chapter 9-10)
- **`dyna.py`** - Dyna-Q and prioritized sweeping (Chapter 8)
- **Examples:**
  - `cliff_walking.py` - Cliff walking (Example 6.6)
  - `windy_grid_world.py` - Windy grid world (Example 6.5)
  - `double_learning.py` - Double Q-learning comparison (Example 6.7)
  - `dyna_maze.py` - Dyna maze (Example 8.1-8.3)
  - `mountain_car.py` - Mountain car with function approximation (Example 10.1)
  - `td_stats.py` - Performance comparison across TD methods

## Usage

```python
from rl.dp.dp import DP
from rl.mc.mc import MC
from rl.td.td import TD, Method
from rl.td.dyna import Dyna
```

Each example can be run standalone from its directory:

```bash
cd dp/examples && python grid_world.py
cd mc/examples && python race_track.py
cd td/examples && python cliff_walking.py
```

## Reference

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
http://incompleteideas.net/book/the-book-2nd.html
