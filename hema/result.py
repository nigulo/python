from dataclasses import dataclass
import numpy as np

@dataclass
class Result:
    battery_power: np.array
    import_power: np.array
    export_power: np.array
    controllable_load_power: np.array
    excess_load_power: np.array
