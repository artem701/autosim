from dataclasses import dataclass

@dataclass
class NCarWatcherUpdate:
    v_avg: float = 0
    u_pos_dt_int: float = 0
