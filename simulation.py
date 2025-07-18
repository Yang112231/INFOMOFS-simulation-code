import argparse
import heapq
import math
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

KW_PER_STATION: int = 6
SECS_PER_HOUR: int = 3600
SECS_PER_DAY:  int = 24 * SECS_PER_HOUR
CABLE_CAPACITY: Dict[int, int] = {0: 1000, **{i: 200 for i in range(1, 10)}}
PARKING_STATIONS: Dict[int, int] = {1: 60, 2: 80, 3: 60, 4: 70, 5: 60, 6: 60, 7: 50}
PARKING_FIRST_CHOICE = np.array([0.15, 0.15, 0.15, 0.20, 0.15, 0.10, 0.10])

# Network topology: parking places and their cable paths
PARKING_PATHS: Dict[int, Tuple[int, ...]] = {
    1: (0, 1, 2),
    2: (0, 1, 3),
    3: (0, 1, 4),
    4: (0, 5),
    5: (0, 5, 7, 8),
    6: (0, 5, 7, 9),
    7: (0, 5, 6),
}

PRICE_LEVELS = [
    (0, 8, 16),   # 00-08h
    (8, 16, 18),  # 08-16h
    (16, 20, 22), # 16-20h
    (20, 24, 20), # 20-24h
]

DATA_FOLDER = "./input/"
OUTPUT_FOLDER = "./output/"
DEF_CSV_KWARGS = dict(sep=";", decimal=",", engine="python")

def normalise_data(csv_name: str, value_col: str, prob_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(f"{DATA_FOLDER}/{csv_name}", **DEF_CSV_KWARGS)
    values = df[value_col].values.astype(float)
    probs = df[prob_col].values.astype(float)
    probs /= probs.sum()
    return values, probs

ARRIVAL_HOURS, ARRIVAL_PROBS = normalise_data(
    "arrival_hours.csv", "Arrival_hour", "Share of charging transactions")
CHARGE_KWH,   CHARGE_PROBS   = normalise_data(
    "charging_volume.csv", "Charging volume [kWh]", "Share of charging transactions")
CONN_HOURS,   CONN_PROBS     = normalise_data(
    "connection_time.csv", "Connection time to charging station [h]", "Share of charging transactions")

SOLAR_DF = pd.read_excel("./input/solar.xlsx", sheet_name="zon")
SOLAR_W = SOLAR_DF["winterAVG"].values.astype(float)
SOLAR_S = SOLAR_DF["zomerAVG"].values.astype(float)
TOTAL_ARR_PER_DAY = 750
ARRIVAL_SHARE = np.array(ARRIVAL_PROBS)
LAMDA_HOURS = ARRIVAL_SHARE * TOTAL_ARR_PER_DAY / SECS_PER_HOUR
LAMDA_MAX = LAMDA_HOURS.max()

# Generate next arrival time
def gen_next_arrival(now: float, rng_arr) -> float:
    while True:
        inter = rng_arr.exponential(1.0 / LAMDA_MAX)
        t_cand = now + inter
        hour = int((t_cand % SECS_PER_DAY) // SECS_PER_HOUR)
        if rng_arr.random() < LAMDA_HOURS[hour] / LAMDA_MAX:
            return t_cand
        now = t_cand

# Data classes and simulation components
@dataclass(order=True)
class Event:
    time: float
    handler: Callable[[], None] = field(compare=False)

class FutureEventsList:
    def __init__(self):
        self.fel: List[Event] = []
    def schedule(self, evt: Event):
        heapq.heappush(self.fel, evt)
    def next(self) -> Optional[Event]:
        return heapq.heappop(self.fel) if self.fel else None

class Vehicle:
    def __init__(self, sim: "Simulation", arrival_time: float):
        self.arrival_time = arrival_time
        self.kwh          = np.random.choice(CHARGE_KWH, p=CHARGE_PROBS)
        self.conn_hours   = np.random.choice(CONN_HOURS, p=CONN_PROBS)
        self.charge_secs  = self.kwh / KW_PER_STATION * SECS_PER_HOUR
        min_conn          = self.charge_secs * 1.4
        self.stay_secs    = max(self.conn_hours * SECS_PER_HOUR, min_conn)
        self.desired_depart_time = self.arrival_time + self.stay_secs
        self.lfs_time = self.desired_depart_time - self.charge_secs
        self.evt_active = False # for preempt
        self.cycle_id = 0 # for preempt
        self.parking: Optional[ParkingPlace] = None
        self.charging_start: Optional[float] = None
        self.charging_end:   Optional[float] = None

    @property
    def delay(self) -> float:
        if self.charging_end is None:
            return 0.0
        return self.charging_end - self.desired_depart_time

class ParkingPlace:
    def __init__(self, pid: int):
        self.pid = pid
        self.capacity = PARKING_STATIONS[pid]
        self.occupied: int = 0
        self.active: List[Vehicle] = []

class Network:
    def __init__(self, scenario: str, rng_solar=None):
        self.load_by_cable: Dict[int, float] = defaultdict(float)
        self.load_history: Dict[int,list[tuple[float,float]]] = defaultdict(list) # (dt, load)
        self.load_ts: Dict[int, list[tuple[float, float]]] = defaultdict(list)
        for cid in CABLE_CAPACITY:
            self.load_ts[cid].append((0.0, 0.0))
        self.rng_solar = rng_solar
        scenario_map = {
            "no_solar":   [],
            "p6p7_w":     [6, 7],
            "p6p7_s":     [6, 7],
            "p1p2p6p7_w": [1, 2, 6, 7],
            "p1p2p6p7_s": [1, 2, 6, 7],
        }
        self.pv_parkings = scenario_map[scenario]
        self.solar_profile = SOLAR_W if "_w" in scenario else SOLAR_S
        self.overload_secs: Dict[int, float]  = defaultdict(float)
        self.blackout_secs: Dict[int, float]  = defaultdict(float)
        self.last_update_time: float = 0.0

    def update_cable_stats(self, now: float):
        dt = now - self.last_update_time
        for cid, load in self.load_by_cable.items():
            self.load_history[cid].append( (dt, abs(load)) ) # update load distribution
            if self.load_ts[cid][-1][1] != abs(load): # update load over time
                    self.load_ts[cid].append((self.last_update_time, abs(load)))

            cap = CABLE_CAPACITY[cid]
            if abs(load) > cap * 1.1:
                self.blackout_secs[cid] += dt
            elif abs(load) > cap:
                self.overload_secs[cid] += dt
        self.last_update_time = now

    def solar_kw(self, park_id: int, now_secs: float) -> float:
        if park_id not in self.pv_parkings:
            return 0.0
        hour = int((now_secs % SECS_PER_DAY) // SECS_PER_HOUR)
        base_kw = 200 * self.solar_profile[hour]
        # Gaussian error 15 % truncated at 0 for realism
        sigma = 0.15 * base_kw
        if sigma == 0:
            return 0.0
        a, b = (0 - base_kw) / sigma, np.inf  # lower-trunc to 0
        return truncnorm(a, b, loc=base_kw, scale=sigma).rvs(random_state=self.rng_solar)

    def add_load(self, park_id: int, kw_delta: float):
        for cid in PARKING_PATHS[park_id]:
            self.load_by_cable[cid] += kw_delta

    def can_add(self, park_id: int, kw_delta: float) -> bool:
        for cid in PARKING_PATHS[park_id]:
            if abs(self.load_by_cable[cid] + kw_delta) > CABLE_CAPACITY[cid]:
                return False
        return True