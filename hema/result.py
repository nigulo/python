from dataclasses import dataclass
import numpy as np

@dataclass
class Result:
    battery: np.array
    buy: np.array
    sell: np.array
    
