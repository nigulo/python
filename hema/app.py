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
            "battery_power": res.battery_power.tolist(),
            "import_power": res.import_power.tolist(),
            "export_power": res.export_power.tolist(),
            "controllable_load_power": res.controllable_load_power.tolist(),
            "excess_load_power": res.excess_load_power.tolist()
            }
        }

    return jsonify(res_dict), 200

def load_data(input):
    data = Data()
    data_dict = input["data"]
    
    if "pv_forecast_power" in data_dict:
        data.pv_forecast_power = np.asarray(data_dict["pv_forecast_power"])
    if "grid_import_price" in data_dict:
        data.grid_import_price = np.asarray(data_dict["grid_import_price"])
    if "grid_export_price" in data_dict:
        data.grid_export_price = np.asarray(data_dict["grid_export_price"])
    if "controllable_load_power" in data_dict:
        controllable_load_power = data_dict["controllable_load_power"]
        if type(controllable_load_power) == list:
            data.controllable_load_power = np.asarray(controllable_load_power)
        else:
            data.controllable_load_power = controllable_load_power
    if "baseline_load_power" in data_dict:
        baseline_load_power = data_dict["baseline_load_power"]
        if type(baseline_load_power) == list:
            data.baseline_load_power = np.asarray(baseline_load_power)
        else:
            data.baseline_load_power = baseline_load_power
    if "battery_current_soc" in data_dict:
        data.battery_current_soc = data_dict["battery_current_soc"]/100
    
    return data

def load_conf(input):
    conf = Conf()
    
    if "conf" not in input:
        return conf

    conf_dict = input["conf"]
    if "battery_capacity" in conf_dict:
        conf.battery_capacity = conf_dict["battery_capacity"]
    if "battery_min_soc" in conf_dict:
        conf.battery_min_soc = conf_dict["battery_min_soc"]/100
    if "battery_max_soc" in conf_dict:
        conf.battery_max_soc = conf_dict["battery_max_soc"]/100
    if "battery_charge_power" in conf_dict:
        conf.battery_charge_power = conf_dict["battery_charge_power"]
    if "battery_discharge_power" in conf_dict:
        conf.battery_discharge_power = conf_dict["battery_discharge_power"]
    if "import_max_power" in conf_dict:
        conf.import_max_power = conf_dict["import_max_power"]
    if "export_max_power" in conf_dict:
        conf.export_max_power = conf_dict["export_max_power"]
    if "max_hours" in conf_dict:
        conf.max_hours = conf_dict["max_hours"]
    if "max_hours_gap" in conf_dict:
        conf.max_hours_gap = conf_dict["max_hours_gap"]
    if "battery_energy_loss" in conf_dict:
        conf.battery_energy_loss = conf_dict["battery_energy_loss"]/100
    if "pv_energy_loss" in conf_dict:
        conf.pv_energy_loss = conf_dict["pv_energy_loss"]/100

    return conf

if __name__ == '__main__':
    app.run(debug=True)