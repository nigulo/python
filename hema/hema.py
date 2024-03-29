import numpy as np
from scipy.optimize import linprog
from data import Data
from conf import Conf
from result import Result

def optimize(data: Data, conf: Conf):
    data_len = len(data.sol)
    if len(data.grid_buy) != data_len or len(data.grid_sell) != data_len or len(data.fixed_cons) != data_len:
        raise "Input data arrays lengths do not match"
        
    if np.any(data.sol < 0):
        raise "Solar energy production cannot be negative"
    if np.any(data.fixed_cons < 0):
        raise "Energy consumption cannot be negative"
        
    if conf.battery_min < 0:
        raise "Minimum battery level must be nonnegative"
    if data.battery_start < conf.battery_min:
        raise "Initial battery level must be greater than or equal to maximum battery level"

    if conf.battery_max is not None:
        if conf.battery_max < 0:
            raise "Maximum battery level must be nonnegative"
        if conf.battery_max < conf.battery_min:
            raise "Maximum battery level must be greater than or equal to minimum battery level"
        if data.battery_start > conf.battery_max:
            raise "Initial battery level must be less than or equal to maximum battery level"
            
    if conf.buy_max is not None and conf.buy_max < 0:
        raise "Maximum grid buy must be nonnegative"
    if conf.sell_max is not None and conf.sell_max < 0:
        raise "Maximum grid sell must be nonnegative"
    if conf.cons_max is not None and conf.cons_max < 0:
        raise "Maximum consumption must be nonnegative"
        
    # x = [battery1, battery2, ...,  buy1, buy2, ..., sell1, sell2, ..., cons1, cons2, ...]
    c = np.concatenate((np.zeros(data_len), data.grid_buy, data.grid_sell, np.zeros(data_len)))

    # battery_min <= battery_start + sum(battery_i) <= battery_max
    #A_ub = np.array([[-1, 0, 0, 0, 0, 0, 0, 0],
    #                 [-1, -1, 0, 0, 0, 0, 0, 0],
    #                 [1, 0, 0, 0, 0, 0, 0, 0],
    #                 [1, 1, 0, 0, 0, 0, 0, 0]])
    
    ones_triangle = np.tril(np.ones((data_len, data_len)))
    ones_triangle_pad = np.pad(ones_triangle, ((0, 0), (0, 3*data_len)))
    
    A_ub = -ones_triangle_pad
    b_ub = np.repeat(data.battery_start-conf.battery_min, data_len)
    if conf.battery_max is not None:
        A_ub = np.concatenate((A_ub, ones_triangle_pad))
        b_ub = np.concatenate((b_ub, np.repeat(conf.battery_max-data.battery_start, data_len)))

    # battery_i + fixed_cons_i + cons_i + sell_i = sol_i + buy_i
    # buy_i*sell_i = 0 (not used)
    #A_eq = np.array([[1, 0, -1, 0, 1, 0, 1, 0],
    #                 [0, 1, 0, -1, 0, 1, 0, 1]])
    A_eq = np.concatenate((np.identity(data_len), 
                           -np.identity(data_len), 
                           np.identity(data_len),
                           np.identity(data_len)), axis=1)
    b_eq = data.sol-data.fixed_cons

    bounds = np.concatenate((np.tile((None, None), (data_len, 1)), 
                             np.tile((0, conf.buy_max), (data_len, 1)),
                             np.tile((0, conf.sell_max), (data_len, 1)),
                             np.tile((0, conf.cons_max), (data_len, 1))))
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ipm')
    if not res.success:
        raise res.message
    return Result(battery=res.x[:data_len], buy=res.x[data_len:2*data_len], sell=res.x[2*data_len:3*data_len], cons=None)