import numpy as np
from scipy.optimize import linprog

battery_min = 0.1
battery_max = 20

battery_start = 0.5

sol = np.array([2.0, 1.0])
grid_buy = np.array([1.0, 2.0])
#grid_sell = np.array([-0.9, -1.9])
consumption = np.array([0.5, 0.4])

num_dev = 2

# x = [battery1, battery2, buy1, buy2]
# sol_i + buy_i = battery_i + consumption_i
# battery_min <= sum(battery_i) <= battery_max

c = np.concatenate((np.zeros(2), grid_buy))
A_ub = np.array([[1, 0, 0, 0],
                  [1, 1, 0, 0],
                  [-1, 0, 0, 0],
                  [-1, -1, 0, 0]])
b_ub = np.concatenate((np.repeat(battery_max-battery_start, 2), np.repeat(battery_start-battery_min, 2)))
A_eq = np.array([[1, 0, -1, 0],
                 [0, 1, 0, -1]])
b_eq = sol - consumption
bounds = [(battery_min, battery_max), (battery_min, battery_max), None, None]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=None, method='highs-ipm')
print(res)