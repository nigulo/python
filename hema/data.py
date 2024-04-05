from dataclasses import dataclass
import numpy as np

@dataclass
class Data:
    pv_power: np.array = None
    grid_buy: np.array = None
    grid_sell: np.array = None
    cons: np.array = None
    fixed_cons: np.array = None
    battery_start: float = 0
    
