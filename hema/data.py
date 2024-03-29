from dataclasses import dataclass
import numpy as np

@dataclass
class Data:
    sol: np.array
    grid_buy: np.array
    grid_sell: np.array
    fixed_cons: np.array
    battery_start: float
    