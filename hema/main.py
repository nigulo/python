import numpy as np
from datetime import datetime

from optimizer import optimize
from data import Data
from conf import Conf

if __name__ == '__main__':
    pv_power = np.loadtxt("solar_forecast.csv", 
                     delimiter=",", 
                     skiprows=1, 
                     usecols=(1, 6), 
                     dtype=object, 
                     converters={1: lambda x: np.datetime64(datetime.strptime(x.decode("utf-8")[1:-1], "%Y-%m-%d %H:%M:%S")),
                                 6: np.float64})
        
    price = np.loadtxt("prices.csv", 
                     delimiter=",", 
                     skiprows=1, 
                     usecols=(2, 3), 
                     dtype=object, 
                     converters={2: lambda x: float(x.decode("utf-8")[1:-1]),
                                 3: lambda x: np.datetime64(datetime.strptime(x.decode("utf-8")[1:-1], "%Y-%m-%d %H:%M:%S"))})
    
    min_time = max(np.min(pv_power[:, 0]), np.min(price[:, 1]))
    max_time = min(np.max(pv_power[:, 0]), np.max(price[:, 1]))
    min_time_h = min_time.astype('datetime64[h]')
    max_time_h = max_time.astype('datetime64[h]')
    if (min_time_h < min_time):
        min_time = min_time_h + np.timedelta64(1, 'h')
    else:
        min_time = min_time_h
    if (max_time_h > max_time):
        max_time = max_time_h - np.timedelta64(1, 'h')
    else:
        max_time = max_time_h
    assert(min_time < max_time)

    pv_power = pv_power[pv_power[:,0] >= min_time]
    pv_power = pv_power[pv_power[:,0] <= max_time]
    price = price[price[:,1] >= min_time]
    price = price[price[:,1] <= max_time]
    
    times_list = []
    pv_power_list = []
    price_list = []
    minutes = np.arange((pv_power[-1,0] - pv_power[0,0]).item().total_seconds()//60, dtype=int, step=15)
    
    j = 0
    k = 0
    for i in range(len(minutes)):
        time = pv_power[0, 0] + np.timedelta64(minutes[i], 'm')
        can_add = len(price_list) > 0
        if price[k, 1] == time:
            price_list.append(price[k])
            k += 1
        elif can_add:
            price_list.append([price_list[-1][0], time])
        can_add = len(price_list) > 0
        if pv_power[j, 0] == time:
            if can_add:
                pv_power_list.append(pv_power[j])
            j += 1
        elif can_add:
            pv_power_list.append([time, 0.])

    pv_power = np.asarray(pv_power_list)
    price = np.asarray(price_list)

    avg_n = 4
    period = 1
    if avg_n > 1:
        pv_power_list = []
        price_list = []
        for i in range(0, len(pv_power), avg_n):
            pv_power_list.append([pv_power[i, 0], np.sum(pv_power[i:i+4, 1])])
        pv_power = np.asarray(pv_power_list)
        price = np.asarray(price[::avg_n])
    
    n_days = 1
    n = n_days*96//avg_n
    data = Data(pv_power=pv_power[:n, 1], 
                grid_buy=price[:n, 0], 
                grid_sell=np.zeros(n),
                fixed_cons=np.ones(n)*800, 
                battery_start=15*1000/2)
    conf = Conf(battery_min= 3*1000,
                battery_max=15*1000,
                battery_charging=5000*period,
                battery_discharging=7000*period,
                buy_max=16*220*3*period,
                sell_max=10*1000*period,
                cons_off_total=6*n_days,
                cons_max_gap=2)

    print(data)
    print(conf)

    res = optimize(data, conf)

    output = np.vstack((np.datetime_as_string(pv_power[:n, 0].astype("datetime64[ns]"), unit='m'),
                        data.pv_power,
                        data.grid_buy,
                        (res.buy-res.sell), 
                        data.fixed_cons,
                        (np.cumsum(res.battery) + data.battery_start)/conf.battery_max*100), dtype=(object)).T
    np.savetxt("output.csv", output, delimiter=",", fmt=("%s", "%.0f", "%.2f", "%.0f", "%.0f", "%.0f"), header="time,PV power,price,grid buy,consumption,battery SOC")
