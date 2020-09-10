import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import gym
import numpy as np
import pandas as pd
import plotly.express as px
from gym import wrappers
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

from src.bc_model import *
from src.consts import *
from src.driving_data import *
from src.driving_utils import *
from src.genetic import main, run_episode
from src.rl import *
from src.utils import load_pkl, print_step, save_pkl

"""
Car Racing
Actions: [steer, gas, break] (floats)
          steer (left) -1  0 1 (right)
          gas 
          break
Observations: 
    96x96x3 image
    greens: (102, 204, 102), (104, 229, 104)
    grey: (102-107, 102-107, 102-107)
"""

# human_race()
