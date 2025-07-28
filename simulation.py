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
SECS_PER_DAY: int = 24 * SECS_PER_HOUR
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
    (0, 8, 16),  # 00-08h
    (8, 16, 18),  # 08-16h
    (16, 20, 22),  # 16-20h
    (20, 24, 20),  # 20-24h
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
CHARGE_KWH, CHARGE_PROBS = normalise_data(
    "charging_volume.csv", "Charging volume [kWh]", "Share of charging transactions")
CONN_HOURS, CONN_PROBS = normalise_data(
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
        self.kwh = np.random.choice(CHARGE_KWH, p=CHARGE_PROBS)
        self.conn_hours = np.random.choice(CONN_HOURS, p=CONN_PROBS)
        self.charge_secs = self.kwh / KW_PER_STATION * SECS_PER_HOUR
        min_conn = self.charge_secs * 1.4
        self.stay_secs = max(self.conn_hours * SECS_PER_HOUR, min_conn)
        self.desired_depart_time = self.arrival_time + self.stay_secs
        self.lfs_time = self.desired_depart_time - self.charge_secs
        self.evt_active = False  # for preempt
        self.cycle_id = 0  # for preempt
        self.parking: Optional[ParkingPlace] = None
        self.charging_start: Optional[float] = None
        self.charging_end: Optional[float] = None

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
        self.load_history: Dict[int, list[tuple[float, float]]] = defaultdict(list)  # (dt, load)
        self.load_ts: Dict[int, list[tuple[float, float]]] = defaultdict(list)
        for cid in CABLE_CAPACITY:
            self.load_ts[cid].append((0.0, 0.0))
        self.rng_solar = rng_solar
        scenario_map = {
            "no_solar": [],
            "p6p7_w": [6, 7],
            "p6p7_s": [6, 7],
            "p1p2p6p7_w": [1, 2, 6, 7],
            "p1p2p6p7_s": [1, 2, 6, 7],
        }
        self.pv_parkings = scenario_map[scenario]
        self.solar_profile = SOLAR_W if "_w" in scenario else SOLAR_S
        self.overload_secs: Dict[int, float] = defaultdict(float)
        self.blackout_secs: Dict[int, float] = defaultdict(float)
        self.last_update_time: float = 0.0

    def update_cable_stats(self, now: float):
        dt = now - self.last_update_time
        for cid, load in self.load_by_cable.items():
            self.load_history[cid].append((dt, abs(load)))  # update load distribution
            if self.load_ts[cid][-1][1] != abs(load):  # update load over time
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


class Aggregator:
    def __init__(self, sim: "Simulation", parking_places: Dict[int, ParkingPlace]):
        self.sim = sim
        self.pp = parking_places

    def on_vehicle_arrival(self, v: Vehicle):
        raise NotImplementedError

    def on_vehicle_depart(self, park_id: int):
        raise NotImplementedError


class BaseStrategy(Aggregator):
    def on_vehicle_arrival(self, v: Vehicle):
        chosen_park = self.sim.select_parking()
        attempts = 1
        while attempts <= 3:
            park = self.pp[chosen_park]
            if park.occupied < park.capacity:
                park.occupied += 1
                v.parking = park
                self.sim.start_charging(v, chosen_park, max(self.sim.now, v.arrival_time))
                return
            chosen_park = self.sim.select_parking(exclude={chosen_park})
            attempts += 1
        self.sim.stats['non_served'] += 1

    def on_vehicle_depart(self, park_id: int):
        park = self.pp[park_id]
        park.occupied -= 1


class PriceDrivenStrategy(BaseStrategy):
    def on_vehicle_arrival(self, v: Vehicle):
        chosen_park = self.sim.select_parking()
        attempts = 1
        while attempts <= 3:
            park = self.pp[chosen_park]
            if park.occupied < park.capacity:
                park.occupied += 1
                v.parking = park
                best_start = self.get_best_start_time(v.arrival_time, v.desired_depart_time, v.charge_secs)
                self.sim.schedule_event(best_start, self.sim.start_charging, v, park.pid, best_start)
                return
            chosen_park = self.sim.select_parking(exclude={chosen_park})
            attempts += 1
        self.sim.stats['non_served'] += 1

    def cost_eur(self, start_secs: float, charge_secs: float) -> float:
        cost = 0.0
        rem = charge_secs
        cur = start_secs
        while rem > 0:
            hour = int((cur % SECS_PER_DAY) // SECS_PER_HOUR)
            price_cents = next(p for (h0, h1, p) in PRICE_LEVELS if h0 <= hour < h1)
            next_boundary = ((hour + 1) * SECS_PER_HOUR + (cur // SECS_PER_DAY) * SECS_PER_DAY)
            dt = min(rem, next_boundary - cur)
            kwh = KW_PER_STATION * dt / SECS_PER_HOUR
            cost += kwh * price_cents / 100
            rem -= dt
            cur += dt
        return cost

    def get_best_start_time(self, arrival_time: float, desired_depart_time: float, charge_secs: float) -> float:
        latest_start = max(arrival_time, desired_depart_time - charge_secs)
        best_start = arrival_time
        best_cost = float('inf')
        cands = {arrival_time, latest_start}
        day0 = int(arrival_time // SECS_PER_DAY)
        day1 = int(latest_start // SECS_PER_DAY)
        for day in range(day0, day1 + 1):
            day_base = day * SECS_PER_DAY
            for h0, _, _ in PRICE_LEVELS:
                t0 = day_base + h0 * SECS_PER_HOUR
                if arrival_time <= t0 <= latest_start:
                    cands.add(t0)

        for t in sorted(cands):
            cost = self.cost_eur(t, charge_secs)
            if cost < best_cost:
                best_cost = cost
                best_start = t

        return best_start


class FCFSStrategy(Aggregator):
    def __init__(self, sim: "Simulation", parking_places):
        super().__init__(sim, parking_places)
        self.wait_q: deque[Vehicle] = deque()

    def on_vehicle_arrival(self, v: Vehicle):
        pid = self.sim.select_parking()
        attempts = 1
        while attempts <= 3:
            park = self.pp[pid]
            if park.occupied < park.capacity:
                park.occupied += 1
                v.parking = park
                self.wait_q.append(v)
                self.try_start_waiting()
                return
            pid = self.sim.select_parking(exclude={pid})
            attempts += 1
        self.sim.stats['non_served'] += 1

    def on_vehicle_depart(self, park_id: int):
        park = self.pp[park_id]
        park.occupied -= 1

    def try_start_waiting(self):  # activate waiting vehicles
        started = True
        while started and self.wait_q:
            started = False
            v = self.wait_q[0]
            pid = v.parking.pid
            if self.sim.net.can_add(pid, KW_PER_STATION):
                self.wait_q.popleft()
                self.sim.start_charging(v, pid, self.sim.now)
                started = True


class ELFSStrategy(FCFSStrategy):
    def try_start_waiting(self):
        while self.wait_q:
            feasible = [
                v for v in self.wait_q
                if self.sim.net.can_add(v.parking.pid, KW_PER_STATION)
            ]
            if not feasible:
                break
            v_sel = min(feasible, key=lambda v: v.lfs_time)
            self.wait_q.remove(v_sel)
            self.sim.start_charging(v_sel, v_sel.parking.pid, self.sim.now)


class PreemptiveStrategy(ELFSStrategy):
    def on_vehicle_preempted(self, v: Vehicle, pid: int):
        charged = self.sim.now - v.charging_start
        v.charge_secs -= charged
        v.lfs_time = v.desired_depart_time - v.charge_secs
        self.wait_q.append(v)

    def handle_solar_drop(self, over_kw: Dict[int, float]):
        still = over_kw.copy()
        victims = sorted(
            (veh for p in self.pp.values() for veh in p.active),
            key=lambda x: x.lfs_time, reverse=True)
        for v in victims:
            if not any(cid in still for cid in PARKING_PATHS[v.parking.pid]):
                continue
            # Interrupt the vehicle charging
            self.sim.handle_charging_end(v, v.parking.pid, v.cycle_id, finished=False)
            for cid in PARKING_PATHS[v.parking.pid]:
                if cid in still:
                    still[cid] -= KW_PER_STATION
                    if still[cid] <= 0:
                        still.pop(cid)
            if not still:
                break  # all cables are safe


# Simulation class to run the DES
class Simulation:
    def __init__(self, strategy_key: str, scenario: str, horizon_days: int, rng_seed: int):
        np.random.seed(rng_seed)
        random.seed(rng_seed)
        self.rng_solar = np.random.default_rng(rng_seed + 100)
        self.rng_arr = np.random.default_rng(rng_seed + 23)
        self.strategy_key = strategy_key
        self.scenario = scenario
        self.horizon_days = horizon_days
        self.now = 0.0
        self.current_day = 0
        self.net = Network(scenario, rng_solar=self.rng_solar)
        self.parking_places: Dict[int, ParkingPlace] = {pid: ParkingPlace(pid) for pid in PARKING_STATIONS}
        strat_map = {
            'base': BaseStrategy,
            'price': PriceDrivenStrategy,
            'fcfs': FCFSStrategy,
            'elfs': ELFSStrategy,
            'preempt': PreemptiveStrategy,
        }
        if strategy_key not in strat_map:
            raise ValueError(f"Unknown strategy {strategy_key}")
        self.aggregator: Aggregator = strat_map[strategy_key](self, self.parking_places)
        self.fel = FutureEventsList()
        self.stats: Dict[str, float] = defaultdict(float)
        self.pv_now: Dict[int, float] = {pid: 0.0 for pid in self.net.pv_parkings}
        self.stats['max_delay'] = 0.0
        self.arrivals_per_day = np.zeros(self.horizon_days, dtype=int)

    def schedule_event(self, time: float, handler: Callable, *args):
        self.fel.schedule(Event(time, lambda: handler(*args)))

    # Event handlers
    def handle_vehicle_arrival(self):
        v = Vehicle(self, self.now)
        self.stats['arrivals'] += 1
        self.arrivals_per_day[self.current_day] += 1
        self.aggregator.on_vehicle_arrival(v)
        t_next = gen_next_arrival(self.now, self.rng_arr)
        if t_next < self.horizon_days * SECS_PER_DAY:
            self.schedule_event(t_next, self.handle_vehicle_arrival)

    def handle_charging_end(self, v: Vehicle, park_id: int, cycle_id: int, finished=True):
        if cycle_id != v.cycle_id:
            return
        if not v.evt_active:
            return
        v.evt_active = False

        park = self.parking_places[park_id]
        if v in park.active:
            park.active.remove(v)

        self.net.update_cable_stats(self.now)
        self.net.add_load(park_id, -KW_PER_STATION)

        if not finished:
            if hasattr(self.aggregator, "on_vehicle_preempted"):
                self.aggregator.on_vehicle_preempted(v, park_id)
        else:
            v.charging_end = self.now
            if hasattr(self.aggregator, "try_start_waiting"):
                self.aggregator.try_start_waiting()
            if v.delay > 0:
                self.stats['delayed'] += 1
                self.stats['delay_time'] += v.delay
                if v.delay > self.stats['max_delay']:
                    self.stats['max_delay'] = v.delay
                self.aggregator.on_vehicle_depart(park_id)
            elif abs(v.delay - 0) < 1e-6:
                self.aggregator.on_vehicle_depart(park_id)
            else:
                self.schedule_event(v.desired_depart_time, self.handle_vehicle_depart, v, park_id)

    def handle_vehicle_depart(self, v: Vehicle, park_id: int):
        self.aggregator.on_vehicle_depart(park_id)

    def handle_solar_update(self):
        self.net.update_cable_stats(self.now)
        for pid in self.net.pv_parkings:
            new_kw = self.net.solar_kw(pid, self.now)
            delta = new_kw - self.pv_now[pid]
            if abs(delta) > 1e-6:
                self.net.add_load(pid, -delta)
                self.pv_now[pid] = new_kw

        over_dict = {
            cid: abs(load) - CABLE_CAPACITY[cid]
            for cid, load in self.net.load_by_cable.items()
            if abs(load) > CABLE_CAPACITY[cid]
        }
        if over_dict and hasattr(self.aggregator, "handle_solar_drop"):
            self.aggregator.handle_solar_drop(over_dict)

        if hasattr(self.aggregator, "try_start_waiting"):
            self.aggregator.try_start_waiting()

        if (self.now + SECS_PER_HOUR < self.horizon_days * SECS_PER_DAY):
            self.schedule_event(self.now + SECS_PER_HOUR, self.handle_solar_update)

    # For three tries to select a parking place
    def select_parking(self, exclude: set[int] | None = None) -> int:
        idxs = list(range(7))
        probs = PARKING_FIRST_CHOICE.copy()
        if exclude:
            for ex in exclude:
                idxs.remove(ex - 1)
                probs[ex - 1] = 0
            probs /= probs.sum()
        return np.random.choice([i + 1 for i in idxs], p=probs[idxs])

    def start_charging(self, v: Vehicle, park_id: int, start_time: float):
        park = self.parking_places[park_id]
        park.active.append(v)
        v.cycle_id += 1
        v.evt_active = True
        v.charging_start = start_time
        self.net.update_cable_stats(start_time)
        self.net.add_load(park_id, KW_PER_STATION)
        finish_time = start_time + v.charge_secs
        self.schedule_event(finish_time, self.handle_charging_end, v, park_id, v.cycle_id)

    def advance(self):
        # schedule first arrival and initial solar update
        t_next = gen_next_arrival(self.now, self.rng_arr)
        self.schedule_event(t_next, self.handle_vehicle_arrival)
        if self.scenario != "no_solar":
            self.schedule_event(self.now, self.handle_solar_update)

        # Main DES loop
        while self.current_day < self.horizon_days:
            evt = self.fel.next()
            if evt is None:
                break
            self.now = evt.time
            self.current_day = int(self.now // SECS_PER_DAY)
            evt.handler()

    # collect results
    def results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cable_metrics = []
        for cid in sorted(CABLE_CAPACITY):
            total_time = self.horizon_days * SECS_PER_DAY
            cable_metrics.append({
                'cable': cid,
                'max_load(kW)': abs(self.net.load_by_cable[cid]),
                'overload(%)': 100 * self.net.overload_secs[cid] / total_time,
                'blackout(%)': 100 * self.net.blackout_secs[cid] / total_time,
            })
        veh_metrics = {
            'arrivals': self.stats['arrivals'],
            'served': self.stats['arrivals'] - self.stats['non_served'],
            'non_served': self.stats['non_served'],
            'non_served(%)': 100 * self.stats['non_served'] / self.stats['arrivals'],
            'delayed': self.stats['delayed'],
            'delayed(%)': 100 * self.stats['delayed'] / self.stats['arrivals'],
            'max_delay(h)': self.stats['max_delay'] / SECS_PER_HOUR if self.stats['delayed'] > 0 else 0.0,
            'avg_delay(h)': self.stats['delay_time'] / (self.stats['delayed'] * SECS_PER_HOUR) if self.stats[
                                                                                                      'delayed'] > 0 else 0.0,
        }
        return pd.DataFrame(cable_metrics), pd.DataFrame([veh_metrics])


# Run experiment and generate results
def run_experiment(args):
    print("\nLoading input and running simulation...")
    cable_collect = []
    veh_collect = []
    all_daily = []
    for run in range(args.runs):
        sim = Simulation(args.strategy, args.scenario, args.horizon, rng_seed=args.seed + run)
        sim.advance()
        df_c, df_v = sim.results()
        cable_collect.append(df_c)
        veh_collect.append(df_v)

        df_day = pd.DataFrame({
            'run': run,
            'day': np.arange(args.horizon),
            'arrivals': sim.arrivals_per_day
        })
        all_daily.append(df_day)

    daily_df = pd.concat(all_daily, ignore_index=True)
    cables = pd.concat(cable_collect).groupby('cable').agg(['mean', 'std'])
    vehicles = pd.concat(veh_collect).agg(['mean', 'std'])

    day_stats = (
        daily_df
        .groupby('day')['arrivals']
        .agg(['mean', 'std'])
        .rename_axis('simulation day')
    )
    # print("\nArrivals per day (across all runs):")
    # print(day_stats.to_string(float_format='%.2f'))

    overall_mean = daily_df['arrivals'].mean()
    overall_std = daily_df['arrivals'].std(ddof=1)
    # print(f"\nOverall arrivals per day: {overall_mean:.2f} +/- {overall_std:.2f}")

    cables.to_csv(f"{OUTPUT_FOLDER}/cables_{args.strategy}_{args.scenario}.csv")
    vehicles.to_csv(f"{OUTPUT_FOLDER}/vehicles_{args.strategy}_{args.scenario}.csv")

    print("\nCable statistics (mean +/- std):\n", cables)
    print("\nVehicle statistics (mean +/- std):\n", vehicles)
    print("\n")

    # average max load per cable plots visualisation
    def plot_peak_load_per_cable():
        data = cables['max_load(kW)']['mean'].reset_index(drop=True)
        plot = data.plot(title="Max load per Cable.  Strategy: %s,  Scenario: %s" % (args.strategy, args.scenario))
        plot.set_xlabel("Cable ID")
        plot.set_ylabel("kW")
        plot.set_xticks(range(len(data)))
        plot.grid(True)
        plt.tight_layout()
        plt.show()

    # plot_peak_load_per_cable() # Uncomment to show the plot

    # Load distribution per cable
    def plot_cable_hist(sim: Simulation, cid: int, bins=20):
        dt_list, load_list = zip(*sim.net.load_history[cid])
        hours = np.array(dt_list) / SECS_PER_HOUR
        plt.figure()
        plt.hist(load_list, bins=bins, weights=hours, edgecolor='black')
        plt.title(f"Cable {cid} load distribution\n"
                  f"Scenario: {sim.scenario}, Strategy: {sim.strategy_key}")
        plt.xlabel("Absolute load (kW)")
        plt.ylabel("Total time (h)")
        plt.tight_layout()
        plt.show()

    # plot_cable_hist(sim, cid=1) # Uncomment to show the plot

    # Load over time per cable
    def plot_cable_timeseries(sim: "Simulation", cid: int, time_unit="h"):  # 'h', 'd', or 's'
        times, loads = zip(*sim.net.load_ts[cid])
        if time_unit == "h":
            times = np.array(times) / SECS_PER_HOUR
            xlabel = "Simulation time (hours)"
        elif time_unit == "d":
            times = np.array(times) / SECS_PER_DAY
            xlabel = "Simulation time (days)"
        else:
            times = np.array(times)
            xlabel = "Simulation time (seconds)"
        times = np.append(times, sim.now / (SECS_PER_HOUR if time_unit == 'h' else
                                            SECS_PER_DAY if time_unit == 'd' else 1))
        loads = np.append(loads, loads[-1])
        plt.figure(figsize=(10, 3))
        plt.step(times, loads, where="post")
        plt.title(f"Cable {cid} Load over time\nStrategy: {sim.strategy_key}, Scenario: {sim.scenario}")
        plt.xlabel(xlabel)
        plt.ylabel("Load (kW)")
        plt.grid(True, alpha=.3)
        plt.tight_layout()
        plt.show()
    # plot_cable_timeseries(sim, cid=1, time_unit="h") # Uncomment to show the plot


def main():
    parser = argparse.ArgumentParser(description="INFOMOFS Smart Charging in Polderwijk")
    parser.add_argument('--strategy', required=True, choices=['base', 'price', 'fcfs', 'elfs', 'preempt'],
                        help='charging control strategy')
    parser.add_argument('--scenario', required=True,
                        choices=['no_solar', 'p6p7_w', 'p6p7_s', 'p1p2p6p7_w', 'p1p2p6p7_s'],
                        help='solar panel scenario')
    parser.add_argument('--runs', type=int, default=10, help='independent replications')
    parser.add_argument('--horizon', type=int, default=30, help='days per replication')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
