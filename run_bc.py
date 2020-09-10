import argparse
import math
import random
import statistics
import time
from collections import (
    defaultdict,
)
from pathlib import (
    Path,
)

import gym
import numpy as np
import pandas as pd
import plotly.express as px
from gym import (
    wrappers,
)
from tensorflow.keras.models import (
    load_model,
    save_model,
)
from tensorflow.keras.utils import (
    plot_model,
)
from tqdm import (
    tqdm,
)

from bc.bc_model import *
from bc.driving_data import *
from common.consts import *
from common.driving_utils import *
from common.utils import (
    load_pkl,
    print_step,
    save_pkl,
)


def run():
    run_bc_model(
        "checkpoints/decent_bc",
        1,
        False,
    )


def check():
    model = load_model(
        "checkpoints/bc"
    )
    (
        imgs,
        features,
        labels,
    ) = (
        load_bc_data()
    )
    n = len(
        imgs
    )
    for _ in range(
        10
    ):
        i = random.randrange(
            0,
            n,
        )
        fig = px.imshow(
            imgs[
                i
            ],
            title=f"{INDEX2STRING[int(labels[i])]}",
        )
        fig.write_image(
            f"imgs/model_validation/{i}.png"
        )
        # fig.show()


def resume():
    model = load_model(
        "checkpoints/bc"
    )
    (
        imgs,
        features,
        labels,
    ) = (
        load_bc_data()
    )
    train_model(
        model,
        imgs,
        features,
        labels,
        40,
    )


# ======= Train =========


def train():
    model = (
        get_bc_model()
    )
    plot_model(
        model,
        "imgs/model.png",
    )
    (
        imgs,
        features,
        labels,
    ) = (
        load_bc_data()
    )
    train_model(
        model,
        imgs,
        features,
        labels,
        40,
    )


# ======= Evaluate =========


def evaluate():
    (
        x,
        y,
    ) = (
        load_bc_data()
    )
    m = load_model(
        "checkpoints/bc/"
    )
    m.evaluate(
        x,
        y,
    )
    pred = m.predict(
        x
    )
    pred = np.argmax(
        pred,
        axis=1,
    )
    print(
        sum(
            y
            == pred
        )
        / len(
            y
        )
    )


# ======= Generate Data =======


def generate():
    human_generate_data(
        False
    )


def gather():
    gather_data_balanced()


def human():
    human_race()


CHOICES = [
    "run",
    "check",
    "resume",
    "train",
    "evaluate",
    "generate",
    "gather",
    "human",
]
CHOICE2FUNC = {
    "run": run,
    "check": check,
    "resume": resume,
    "train": train,
    "evaluate": evaluate,
    "generate": generate,
    "gather": gather,
    "human": human,
}

if (
    __name__
    == "__main__"
):
    parser = (
        argparse.ArgumentParser()
    )
    parser.add_argument(
        "action",
        type=str,
        choices=CHOICES,
    )
    args = (
        parser.parse_args()
    )
    CHOICE2FUNC[
        args.action
    ]()


# TODO simplify data by setting the speed automatically
# TODO get rl agent working!

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
