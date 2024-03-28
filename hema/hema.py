import numpy as np
from scipy.optimize import linprog

battery_min = 0.1
battery_max = 20

battery_start = 0.5

sol = np.array([2.0, 1.0])
grid_buy = np.array([1.0, 2.0])
grid_sell = np.array([-0.9, -1.9])
consumption = np.array([0.5, 0.4])

num_dev = 2

# x = [battery1, battery2, buy1, buy2, sell1, sell2]
c = np.concatenate((np.zeros(2), grid_buy, grid_sell))

# battery_min <= battery_start + sum(battery_i) <= battery_max
# battery_start + sum(battery_i-1) >= sell_i + consumption_i
A_ub = np.array([[1, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0],
                  [-1, 0, 0, 0, 0, 0],
                  [-1, -1, 0, 0, 0, 0],
                  [-1, 0, 0, 0, 0, 1]])
b_ub = np.concatenate((np.repeat(battery_max-battery_start, 2), 
                       np.repeat(battery_start-battery_min, 2),
                       np.array([battery_start - consumption[1]])))

# battery_i + consumption_i + sell_i = sol_i + buy_i
A_eq = np.array([[1, 0, -1, 0, 1, 0],
                 [1, 1, 0, -1, 0, 1]])
b_eq = sol - consumption
bounds = [(battery_min, battery_max), (battery_min, battery_max), (0, None), (0, None), (0, None), (0, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ipm')
print(res)