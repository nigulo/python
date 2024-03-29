from dataclasses import dataclass
import numpy as np

@dataclass
class Conf:
    battery_min: float = 0
    battery_max: float = np.inf
    buy_max: float = np.inf
    sell_max: float = np.inf
    cons_max: float = np.inf