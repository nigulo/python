from dataclasses import dataclass

@dataclass
class Conf:
    battery_min: float = 0
    battery_max: float = None
    battery_charging: float = None
    battery_discharging: float = None
    buy_max: float = None
    sell_max: float = None
