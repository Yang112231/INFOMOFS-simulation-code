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