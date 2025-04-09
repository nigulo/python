from dataclasses import dataclass
import numpy as np

@dataclass
class Data:
    pv_forecast_power: np.array = None
    grid_import_price: np.array = None
    grid_export_price: np.array = None
    controllable_load_power: np.array = None
    baseline_load_power: np.array = None
    battery_current_soc: float = 0
