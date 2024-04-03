import numpy as np
from datetime import datetime

from hema import optimize
from data import Data
from conf import Conf
from result import Result


if __name__ == '__main__':
    sol = np.loadtxt("solar_forecast.csv", 
                     delimiter=",", 
                     skiprows=1, 
                     usecols=(1, 6), 
                     dtype=object, 
                     converters={1: lambda x: np.datetime64(datetime.strptime(x.decode("utf-8")[1:-1], "%Y-%m-%d %H:%M:%S")),
                                 6: np.float64})
    
    print(sol[sol[:,0] <= np.datetime64("2024-03-20 04:15:00")])
    
    price = np.loadtxt("prices.csv", 
                     delimiter=",", 
                     skiprows=1, 
                     usecols=(2, 3), 
                     dtype=object, 
                     converters={2: lambda x: float(x.decode("utf-8")[1:-1]),
                                 3: lambda x: np.datetime64(datetime.strptime(x.decode("utf-8")[1:-1], "%Y-%m-%d %H:%M:%S"))})
    
    min_time = max(np.min(sol[:, 0]), np.min(price[:, 1]))
    max_time = min(np.max(sol[:, 0]), np.max(price[:, 1]))
    print(min_time)
    print(max_time)
    sol = sol[sol[:,0] >= min_time]
    sol = sol[sol[:,0] <= max_time]
    price = price[price[:,1] >= min_time]
    price = price[price[:,1] <= max_time]
    
    print(sol[0], sol[-1])
    print(price[0], price[-1])
    print((sol[-1,0] - sol[0,0]).item().total_seconds()/(3600))
    times_list = []
    sol_list = []
    price_list = []
    minutes = np.arange((sol[-1,0] - sol[0,0]).item().total_seconds()//60, dtype=int, step=15)
    j = 0
    k = 0
    print(sol[0], price[0])
    for i in range(len(minutes)):
        time = sol[0, 0] + np.timedelta64(minutes[i], 'm')
        can_add = len(price_list) > 0
        if price[k, 1] == time:
            price_list.append(price[k])
            k += 1
        elif can_add:
            price_list.append([price_list[-1][0], time])
        can_add = len(price_list) > 0
        if sol[j, 0] == time:
            if can_add:
                sol_list.append(sol[j])
            j += 1
        elif can_add:
            sol_list.append([time, 0.])
        
    sol = np.asarray(sol_list)
    price = np.asarray(price_list)

    data = Data(sol=sol[:96, 1]/1000, 
                grid_buy=price[:96, 0], 
                grid_sell=price[:96, 0], 
                fixed_cons=np.zeros(96), 
                battery_start=0)
    conf = Conf(battery_max=15,
                battery_charging=5/4,
                battery_discharging=7/4,
                buy_max=16*220/1000/4,
                sell_max=10/4)

    res = optimize(data, conf)
    print(res)
    n = 96
    output = np.concatenate((np.datetime_as_string(sol[:n, 0].astype("datetime64[ns]"), unit='m').reshape(n, 1), 
                             (res.buy-res.sell).reshape(n, 1), 
                             np.cumsum(res.battery).reshape(n, 1)), axis=1, dtype=(object))
    print(output)
    np.savetxt("output.csv", output, delimiter=",", fmt=("%s", "%.3f", "%.3f"), header="time,buy(sell),battery")
