import numpy as np
from scipy.optimize import linprog
from data import Data
from conf import Conf
from result import Result

def optimize(data: Data, conf: Conf):
    if data.pv_power is not None:
        data_len = len(data.pv_power)
    elif data.grid_buy is not None:
        data_len = len(data.grid_buy)
    elif data.grid_sell is not None:
        data_len = len(data.grid_sell)
    elif data.cons is not None:
        data_len = len(data.cons)
    elif data.fixed_cons is not None:
        data_len = len(data.fixed_cons)
    else:
        raise Exception("Missing input data")

    if data.pv_power is None:
        data.pv_power = np.zeros(data_len)
    if data.grid_buy is None:
        data.grid_buy = np.zeros(data_len)
    if data.grid_sell is None:
        data.grid_sell = np.zeros(data_len)
    if data.cons is None:
        data.cons = np.zeros(data_len)
    elif np.isscalar(data.cons):
        data.cons = np.repeat(data.cons, data_len)
    if data.fixed_cons is None:
        data.fixed_cons = np.zeros(data_len)
    elif np.isscalar(data.fixed_cons):
        data.fixed_cons = np.repeat(data.fixed_cons, data_len)
    
    if len(data.grid_buy) != data_len or len(data.grid_sell) != data_len or len(data.fixed_cons) != data_len or len(data.cons) != data_len:
        raise Exception("Input data array lengths do not match")
        
    if np.any(data.pv_power < 0):
        raise Exception("Solar energy production cannot be negative")
    if np.any(data.fixed_cons < 0):
        raise Exception("Energy consumption cannot be negative")
        
    if conf.battery_min < 0:
        raise Exception("Minimum battery level must be nonnegative")
    if data.battery_start < conf.battery_min:
        raise Exception("Initial battery level must be greater than or equal to maximum battery level")

    if conf.battery_max is not None:
        if conf.battery_max < 0:
            raise Exception("Maximum battery level must be nonnegative")
        if conf.battery_max < conf.battery_min:
            raise Exception("Maximum battery level must be greater than or equal to minimum battery level")
        if data.battery_start > conf.battery_max:
            raise Exception("Initial battery level must be less than or equal to maximum battery level")
    if conf.battery_charging is not None and conf.battery_charging < 0:
        raise Exception("Battery charging power must be nonnegative")
    if conf.battery_discharging is not None and conf.battery_discharging < 0:
        raise Exception("Battery discharging power must be nonnegative")
    
    if conf.cons_max_gap < 0:
        raise Exception("Consumption maximum gap must be nonnegative")
    if conf.cons_max_gap > data_len:
        raise Exception(f"Consumption maximum gap must be less than {data_len}")
    if conf.cons_off_total < conf.cons_max_gap:
        raise Exception("Consumption total off hours must greater than or equal to consumption maximum gap")
    if conf.cons_off_total > data_len:
        raise Exception(f"Consumption total off hours must be less than {data_len}")
        
    if conf.buy_max is not None and conf.buy_max < 0:
        raise Exception("Maximum grid buy must be nonnegative")
    if conf.sell_max is not None and conf.sell_max < 0:
        raise Exception("Maximum grid sell must be nonnegative")
               
    # x = [battery1, battery2, ...,  buy1, buy2, ..., sell1, sell2, ..., excess_cons1, excess_cons2, ..., cons1, cons2, ...]
    # Set -eps as a weight for battery to prefer charging over consuming in the case of negative grid buy
    c = np.concatenate((-np.ones(data_len)*1e-6, 
                        data.grid_buy, 
                        -data.grid_sell, 
                        np.zeros(2*data_len)))
    
    A_ub, b_ub = get_ub(data, conf)
    A_eq, b_eq = get_eq(data, conf)
    bounds = get_bounds(data, conf)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ipm')

    if not res.success:
        raise Exception(res.message)
    
    res_cons = res.x[4*data_len:]
    if np.max(data.cons) > 0:
        cons_diff = data.cons - res_cons
        idx = cons_diff < data.cons/2
        res_cons[idx] = data.cons[idx]
        res_cons[res_cons < data.cons] = 0
        data.fixed_cons += res_cons
        data.cons = np.zeros(data_len)
        
        A_ub, b_ub = get_ub(data, conf)
        A_eq, b_eq = get_eq(data, conf)
        bounds = get_bounds(data, conf)
    
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ipm')
    
        if not res.success:
            raise Exception(res.message)

    return Result(battery=res.x[:data_len], 
                  buy=res.x[data_len:2*data_len], 
                  sell=res.x[2*data_len:3*data_len], 
                  excess_cons=res.x[3*data_len:4*data_len],
                  cons=res_cons)

def get_ub(data: Data, conf: Conf):
    data_len = len(data.pv_power)
    
    # battery_min <= battery_start + sum(battery_i) <= battery_max
    #A_ub = np.array([[-1, 0, 0, 0, 0, 0, 0, 0],
    #                 [-1, -1, 0, 0, 0, 0, 0, 0],
    #                 [1, 0, 0, 0, 0, 0, 0, 0],
    #                 [1, 1, 0, 0, 0, 0, 0, 0]])

    ones_triangle = np.tril(np.ones((data_len, data_len)))
    ones_triangle_pad = np.pad(ones_triangle, ((0, 0), (0, 4*data_len)))

    A_ub = -ones_triangle_pad
    b_ub = np.repeat(data.battery_start-conf.battery_min, data_len)
    if conf.battery_max is not None:
        A_ub = np.concatenate((A_ub, ones_triangle_pad))
        b_ub = np.concatenate((b_ub, np.repeat(conf.battery_max-data.battery_start, data_len)))
    if np.max(data.cons) > 0:
        if (conf.cons_max_gap > 0):
            # In each interval of length cons_max_gap+1 there must be positive consumption
            # sum(cons_i, ... cons_{i+cons_max_gap+1}) >= cons, where 0 <= i = n-cons_max_gap
            for i in range(data_len-conf.cons_max_gap):
                A_ub = np.concatenate((A_ub, np.pad(-np.ones(conf.cons_max_gap+1), (4*data_len+i, data_len-i-conf.cons_max_gap-1)).reshape(1, -1)))
                b_ub = np.concatenate((b_ub, np.array([-np.min(data.cons[i:i+conf.cons_max_gap+1])])))
        # Total off hours must be <= cons_off_total
        # sum(cons_i, ... cons_n >= cons*(n-cons_off_total)
        A_ub = np.concatenate((A_ub, (np.pad(-np.ones(data_len), (4*data_len, 0))).reshape(1, -1)))
        b_ub = np.concatenate((b_ub, np.array([-np.sum(np.sort(data.cons)[:data_len-conf.cons_off_total])])))
    
    return A_ub, b_ub

def get_eq(data: Data, conf: Conf):
    data_len = len(data.pv_power)

    # battery_i + fixed_cons_i + cons_i + sell_i = pv_power_i + buy_i
    # buy_i*sell_i = 0 (not used)
    #A_eq = np.array([[1, 0, -1, 0, 1, 0, 1, 0, 1, 0],
    #                 [0, 1, 0, -1, 0, 1, 0, 1, 0, 1]])
    A_eq = np.concatenate((np.identity(data_len), 
                           -np.identity(data_len), 
                           np.identity(data_len),
                           np.identity(data_len),
                           np.identity(data_len)), axis=1)
    b_eq = data.pv_power-data.fixed_cons
    
    return A_eq, b_eq

def get_bounds(data: Data, conf: Conf):
    data_len = len(data.pv_power)

    if conf.battery_discharging is not None:
        discharging = -conf.battery_discharging
    else:
        discharging = None
    bounds = np.concatenate((np.tile((discharging, conf.battery_charging), (data_len, 1)), 
                             np.tile((0, conf.buy_max), (data_len, 1)),
                             np.tile((0, conf.sell_max), (data_len, 1)),
                             np.tile((0, None), (data_len, 1)),
                             np.vstack((np.zeros(data_len), data.cons)).T))

    return bounds