from dataclasses import dataclass
import numpy as np

@dataclass
class Conf:
    battery_min: float = 0
    battery_max: float = None