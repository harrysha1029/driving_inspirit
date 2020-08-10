import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

import gym
import numpy as np
import pandas as pd
import plotly.express as px
from gym import wrappers
from src.bc_model import *
from src.driving_data import *
from src.driving_utils import *
from src.consts import *
from src.genetic import main, run_episode
from src.utils import load_pkl, print_step, save_pkl
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils import plot_model

# TODO simplify data by setting the speed automatically
# TODO get rl agent working!

# run_bc_model('checkpoints/bc', 1, False)

# =======  Check model ========

# model = load_model("checkpoints/bc")
# imgs, features, labels = load_bc_data()
# n = len(imgs)
# for _ in range(10):
#     i = random.randrange(0,n)
#     fig = px.imshow(imgs[i], title=f'{INDEX2STRING[int(labels[i])]}')
#     fig.write_image(f"imgs/model_validation/{i}.png")
#     # fig.show()


# ======= Resume Training ========

# model = load_model("checkpoints/bc")
# imgs, features, labels = load_bc_data()
# train_model(model, imgs, features, labels)

# ======= Train =========

model = get_bc_model()
plot_model(model, 'imgs/model.png')
imgs, features, labels = load_bc_data()
train_model(model, imgs, features, labels, 40)

# ======= Evaluate =========

# x, y = load_bc_data()
# m = load_model('checkpoints/bc/')
# m.evaluate(x, y)
# pred = m.predict(x)
# pred = np.argmax(pred, axis=1)
# print(sum(y==pred)/len(y))

# ======= Generate Data =======

# human_generate_data(False)
# gather_data_balanced()

# env = gym.make("CarRacing-v0")
# env.reset()
# print(env.action_space)

"""
Car Racing
Actions: [steer, gas, break] (floats)
          steer (left) -1  0 1 (right)
          gas 
          break
Observations: 
    92x92x3 image
    greens: (102, 204, 102), (104, 229, 104)
    grey: (102-107, 102-107, 102-107)
"""

# human_race()
