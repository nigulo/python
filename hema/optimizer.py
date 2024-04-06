import numpy as np
from scipy.optimize import linprog
from data import Data
from conf import Conf
from result import Result

def optimize(data: Data, conf: Conf):
    _validate_data(data)    
    _validate_conf(data, conf)    
                   
    data_len = _data_len(data)

    # x = [battery1, battery2, ...,  buy1, buy2, ..., sell1, sell2, ..., excess_cons1, excess_cons2, ..., controllable_load_power1, controllable_load_power2, ...]
    # Set -eps as a weight for battery to prefer charging over consuming in the case of negative grid buy
    c = np.concatenate((-np.ones(data_len)*1e-6, 
                        data.grid_import_price, 
                        -data.grid_export_price, 
                        np.zeros(2*data_len)))
    
    A_ub, b_ub = _get_ub(data, conf)
    A_eq, b_eq = _get_eq(data, conf)
    bounds = _get_bounds(data, conf)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ipm')

    if not res.success:
        raise Exception(res.message)
    
    res_controllable_load_power = res.x[4*data_len:]
    if np.max(data.controllable_load_power) > 0:
        controllable_load_power_diff = data.controllable_load_power - res_controllable_load_power
        idx = controllable_load_power_diff < data.controllable_load_power/2
        res_controllable_load_power[idx] = data.controllable_load_power[idx]
        res_controllable_load_power[res_controllable_load_power < data.controllable_load_power] = 0
        data.baseline_load_power += res_controllable_load_power
        data.controllable_load_power = np.zeros(data_len)
        
        A_ub, b_ub = _get_ub(data, conf)
        A_eq, b_eq = _get_eq(data, conf)
        bounds = _get_bounds(data, conf)
    
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ipm')
    
        if not res.success:
            raise Exception(res.message)

    return Result(battery=res.x[:data_len], 
                  buy=res.x[data_len:2*data_len], 
                  sell=res.x[2*data_len:3*data_len], 
                  excess_cons=res.x[3*data_len:4*data_len],
                  controllable_load_power=res_controllable_load_power)

def _validate_data(data: Data):
    if data.pv_forecast_power is not None:
        data_len = len(data.pv_forecast_power)
    elif data.grid_import_price is not None:
        data_len = len(data.grid_import_price)
    elif data.grid_export_price is not None:
        data_len = len(data.grid_export_price)
    elif data.controllable_load_power is not None:
        data_len = len(data.controllable_load_power)
    elif data.baseline_load_power is not None:
        data_len = len(data.baseline_load_power)
    else:
        raise Exception("Missing input data")

    if data.pv_forecast_power is None:
        data.pv_forecast_power = np.zeros(data_len)
    if data.grid_import_price is None:
        data.grid_import_price = np.zeros(data_len)
    if data.grid_export_price is None:
        data.grid_export_price = np.zeros(data_len)
    if data.controllable_load_power is None:
        data.controllable_load_power = np.zeros(data_len)
    elif np.isscalar(data.controllable_load_power):
        data.controllable_load_power = np.repeat(data.controllable_load_power, data_len)
    if data.baseline_load_power is None:
        data.baseline_load_power = np.zeros(data_len)
    elif np.isscalar(data.baseline_load_power):
        data.baseline_load_power = np.repeat(data.baseline_load_power, data_len)
    
    if len(data.grid_import_price) != data_len or len(data.grid_export_price) != data_len or len(data.baseline_load_power) != data_len or len(data.controllable_load_power) != data_len:
        raise Exception("Input data array lengths do not match")
        
    if np.any(data.pv_forecast_power < 0):
        raise Exception("Solar power prediction cannot be negative")
    if np.any(data.baseline_load_power < 0):
        raise Exception("Baseline load power cannot be negative")

def _validate_conf(data: Data, conf: Conf):
    data_len = _data_len(data)
    
    if conf.battery_min_soc < 0:
        raise Exception("Minimum battery SOC must be nonnegative")
    if conf.battery_capacity is not None and conf.battery_capacity < 0:
        raise Exception("Battery capacity must be nonnegative")
    if conf.battery_capacity is None and (data.battery_current_soc > 0 or conf.battery_min_soc > 0 or conf.battery_max_soc is not None):
        raise Exception("Battery capacity must be defined")
    if data.battery_current_soc < conf.battery_min_soc:
        raise Exception("Current battery SOC must be greater than or equal to minimum battery SOC")

    if conf.battery_max_soc is not None:
        if conf.battery_max_soc < 0:
            raise Exception("Maximum battery SOC must be nonnegative")
        if conf.battery_max_soc < conf.battery_min_soc:
            raise Exception("Maximum battery SOC must be greater than or equal to minimum battery SOC")
        if data.battery_current_soc > conf.battery_max_soc:
            raise Exception("Current battery SOC must be less than or equal to maximum battery SOC")
    if conf.battery_charge_power is not None and conf.battery_charge_power < 0:
        raise Exception("Battery charging power must be nonnegative")
    if conf.battery_discharge_power is not None and conf.battery_discharge_power < 0:
        raise Exception("Battery discharging power must be nonnegative")

    if conf.battery_energy_loss < 0:
        raise Exception("Battery energy loss must be nonnegative")
    if conf.pv_energy_loss < 0:
        raise Exception("PV energy loss must be nonnegative")
    
    if conf.max_hours_gap < 0:
        raise Exception("Consumption maximum gap must be nonnegative")
    if conf.max_hours_gap > data_len:
        raise Exception(f"Consumption maximum gap must be less than {data_len}")
    if conf.max_hours+conf.max_hours_gap > data_len:
        raise Exception(f"Max hours plus max hours gap must less than or equal to {data_len}")
    if conf.max_hours > data_len:
        raise Exception(f"Max hours must be less than or equal to {data_len}")
        
    if conf.import_max_power is not None and conf.import_max_power < 0:
        raise Exception("Maximum import power must be nonnegative")
    if conf.export_max_power is not None and conf.export_max_power < 0:
        raise Exception("Maximum export power must be nonnegative")

def _get_ub(data: Data, conf: Conf):
    data_len = _data_len(data)
    
    battery_start = 0
    if data.battery_current_soc > 0:
        battery_start = conf.battery_capacity*data.battery_current_soc
    battery_min = 0
    if conf.battery_min_soc > 0:
        battery_min = conf.battery_capacity*conf.battery_min_soc
    battery_max = None
    if conf.battery_max_soc is not None:
        battery_max = conf.battery_capacity*conf.battery_max_soc
    elif conf.battery_capacity is not None:
        battery_max = conf.battery_capacity
        
    # battery_min <= battery_start + sum(battery_i) <= battery_max
    #A_ub = np.array([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

    ones_triangle = np.tril(np.ones((data_len, data_len)))
    ones_triangle_pad = np.pad(ones_triangle, ((0, 0), (0, 4*data_len)))

    A_ub = -ones_triangle_pad
    b_ub = np.repeat(battery_start-battery_min, data_len)
    if battery_max is not None:
        A_ub = np.concatenate((A_ub, ones_triangle_pad))
        b_ub = np.concatenate((b_ub, np.repeat(battery_max-battery_start, data_len)))
    if np.max(data.controllable_load_power) > 0:
        if (conf.max_hours_gap > 0):
            # In each interval of length max_hours_gap+1 there must be positive consumption
            # sum(controllable_load_power_i, ... controllable_load_power_{i+max_hours_gap+1}) >= cons, where 0 <= i = n-max_hours_gap
            for i in range(data_len-conf.max_hours_gap):
                A_ub = np.concatenate((A_ub, np.pad(-np.ones(conf.max_hours_gap+1), (4*data_len+i, data_len-i-conf.max_hours_gap-1)).reshape(1, -1)))
                b_ub = np.concatenate((b_ub, np.array([-np.min(data.controllable_load_power[i:i+conf.max_hours_gap+1])])))
        # Total on hours must be >= max_hours
        A_ub = np.concatenate((A_ub, (np.pad(-np.ones(data_len), (4*data_len, 0))).reshape(1, -1)))
        b_ub = np.concatenate((b_ub, np.array([-np.sum(np.sort(data.controllable_load_power)[:conf.max_hours])])))
    
    return A_ub, b_ub

def _get_eq(data: Data, conf: Conf):
    data_len = _data_len(data)
    
    # battery_i + baseline_load_power_i + excess_cons + controllable_load_power_i + sell_i = pv_forecast_power_i + buy_i
    #A_eq = np.array([[1, 0, -1, 0, 1, 0, 1, 0, 1, 0],
    #                 [0, 1, 0, -1, 0, 1, 0, 1, 0, 1]])
    id = np.identity(data_len)
    battery_loss = 1-conf.battery_energy_loss
    id_with_loss = id/battery_loss
    A_eq = np.hstack((id, -id_with_loss, id_with_loss, id_with_loss, id_with_loss))
    b_eq = data.pv_forecast_power*(1-conf.pv_energy_loss)*battery_loss - data.baseline_load_power/battery_loss
    
    return A_eq, b_eq

def _get_bounds(data: Data, conf: Conf):
    data_len = _data_len(data)

    if conf.battery_discharge_power is not None:
        discharge_power = -conf.battery_discharge_power
    else:
        discharge_power = None
    bounds = np.concatenate((np.tile((discharge_power, conf.battery_charge_power), (data_len, 1)), 
                             np.tile((0, conf.import_max_power), (data_len, 1)),
                             np.tile((0, conf.export_max_power), (data_len, 1)),
                             np.tile((0, None), (data_len, 1)),
                             np.vstack((np.zeros(data_len), data.controllable_load_power)).T))

    return bounds

def _data_len(data: Data):
    return len(data.pv_forecast_power)