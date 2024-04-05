from flask import Flask, jsonify, request
import traceback
import numpy as np

from optimizer import optimize
from data import Data
from conf import Conf

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input = request.get_json()
    app.logger.info(input)
    
    if "data" not in input:
        return "No data provided", 400
        
    try:
        data = load_data(input)
    except Exception as ex:
        app.logger.error(traceback.format_exc())
        return f"Error loading data: {ex}", 400

    try:
        conf = load_conf(input)
    except Exception as ex:
        app.logger.error(traceback.format_exc())
        return f"Error loading conf: {ex}", 400
                
    try:
        res = optimize(data, conf)
    except Exception as ex:
        app.logger.error(traceback.format_exc())
        return f"Error in optimization: {ex}", 400
    
    res_dict = {
        "result": {
            "battery": res.battery.tolist(),
            "buy": res.buy.tolist(),
            "sell": res.sell.tolist(),
            "cons": res.cons.tolist(),
            "excess_cons": res.excess_cons.tolist()
            }
        }

    return jsonify(res_dict), 200

def load_data(input):
    data = Data()
    data_dict = input["data"]
    
    if "pv_power" in data_dict:
        data.pv_power = np.asarray(data_dict["pv_power"])
    if "grid_buy" in data_dict:
        data.grid_buy = np.asarray(data_dict["grid_buy"])
    if "grid_sell" in data_dict:
        data.grid_sell = np.asarray(data_dict["grid_sell"])
    if "cons" in data_dict:
        cons = data_dict["cons"]
        if type(cons) == list:
            data.cons = np.asarray(cons)
        else:
            data.cons = cons
    if "fixed_cons" in data_dict:
        fixed_cons = data_dict["fixed_cons"]
        if type(fixed_cons) == list:
            data.fixed_cons = np.asarray(fixed_cons)
        else:
            data.fixed_cons = fixed_cons
    if "battery_start" in data_dict:
        data.battery_start = data_dict["battery_start"]
    
    return data

def load_conf(input):
    conf = Conf()
    
    if "conf" not in input:
        return conf

    conf_dict = input["conf"]
    if "battery_min" in conf_dict:
        conf.battery_min = conf_dict["battery_min"]
    if "battery_max" in conf_dict:
        conf.battery_max = conf_dict["battery_max"]
    if "battery_charging" in conf_dict:
        conf.battery_charging = conf_dict["battery_charging"]
    if "battery_discharging" in conf_dict:
        conf.battery_discharging = conf_dict["battery_discharging"]
    if "buy_max" in conf_dict:
        conf.buy_max = conf_dict["buy_max"]
    if "sell_max" in conf_dict:
        conf.sell_max = conf_dict["sell_max"]
    if "cons_off_total" in conf_dict:
        conf.cons_off_total = conf_dict["cons_off_total"]
    if "cons_max_gap" in conf_dict:
        conf.cons_max_gap = conf_dict["cons_max_gap"]

    return conf

if __name__ == '__main__':
    app.run(debug=True)