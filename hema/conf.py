from dataclasses import dataclass

@dataclass
class Conf:
    battery_min: float = 0
    battery_max: float = None
    battery_charging: float = None
    battery_discharging: float = None
    buy_max: float = None
    sell_max: float = None
    cons_power: float = 0
    cons_off_total: int = 0
    cons_max_gap: int = 0
