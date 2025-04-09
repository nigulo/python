from dataclasses import dataclass

@dataclass
class Conf:
    battery_capacity: float = None
    battery_min_soc: float = 0
    battery_max_soc: float = None
    battery_charge_power: float = None
    battery_discharge_power: float = None
    import_max_power: float = None
    export_max_power: float = None
    max_hours: int = 0
    max_hours_gap: int = 0
    battery_energy_loss: float = 0
    pv_energy_loss: float = 0
