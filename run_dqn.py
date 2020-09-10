import os
import pickle
import random
import sys

import gym
import numpy as np
import plotly.express as px
from gym.wrappers import Monitor
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
    MaxPool2D,
    Permute,
    Reshape,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from common.consts import GRASS, INDEX2ACTIONS, ROAD
from common.driving_utils import get_features_from_env
from common.utils import load_json
from rl.agents.dqn import DQNAgent
from rl.callbacks import Callback, FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy, Policy
from rl.processors import Processor


def save_pkl(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


ENV_NAME = "CarRacing-v0"
INPUT_SHAPE = (96, 96, 3)
WINDOW_LENGTH = 3
VIDEO_INTERVAL = 25  # in episodes
TRAIN_STEPS = 175000
ANNEAL_STEPS = 100000


class RacingProcessor(Processor):
    def __init__(self, env):
        self.env = env
        self.count = 0

    def process_action(self, action):
        return INDEX2ACTIONS[action]


def start_policy(state):
    return 1


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, grass_penalty, still_penalty):
        super().__init__(env)
        self.grass_penalty = grass_penalty
        self.still_penalty = still_penalty

    def reward(self, rew):
        left_col = tuple(self.env.state[65, 50, :])
        right_col = tuple(self.env.state[65, 45, :])
        if rew < 0:
            rew = -self.still_penalty
        if left_col not in ROAD or right_col not in ROAD:
            rew = -self.grass_penalty
        return rew


# Get the environment and extract the number of actions.


def get_driving_env_dqn(name, grass_penalty, still_penalty):
    env = Monitor(
        RewardWrapper(gym.make(ENV_NAME), grass_penalty, still_penalty),
        f"/home/harry/Documents/videos/{name}",
        lambda x: x % VIDEO_INTERVAL == 0,
        force=True,
    )
    return env


# np.random.seed(123)
# env.seed(123)
NB_ACTIONS = 4

# Next, we build a very simple model.
model = Sequential(
    [
        Permute((2, 3, 1, 4), input_shape=(WINDOW_LENGTH, 96, 96, 3)),
        Reshape((96, 96, -1)),
        Lambda(lambda x: x / 255.0),
        MaxPool2D(2),
        Conv2D(32, 3, activation="relu"),
        MaxPool2D(2),
        # Dropout(0.2),
        Conv2D(64, 3, activation="relu"),
        MaxPool2D(2),
        # Dropout(0.2),
        Conv2D(64, 3, activation="relu"),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(NB_ACTIONS),
    ]
)
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)
# policy = BoltzmannQPolicy()
annealed_policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr="eps",
    value_max=1,
    value_min=0.1,
    value_test=0.05,
    nb_steps=ANNEAL_STEPS,
)
eps_greedy_policy = EpsGreedyQPolicy(0.1)


test_policy = EpsGreedyQPolicy(0.05)


def get_dqn(env, policy):
    dqn = DQNAgent(
        model=model,
        nb_actions=NB_ACTIONS,
        policy=policy,
        memory=memory,
        processor=RacingProcessor(env),
        nb_steps_warmup=100,
        gamma=0.95,
        target_model_update=10000,
        train_interval=4,
        # delta_clip=1.0,
        test_policy=SpeedLimitPolicy(env, 3000),
        enable_dueling_network=True,
    )

    dqn.compile(Adam(learning_rate=1e-4), metrics=["mae"])
    return dqn


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.


def load(dqn, fname):
    dqn.load_weights(fname)
    return dqn


def train(env, dqn, name, action_rep):
    log_path = f"/home/harry/Documents/driving_models/{name}.log"
    checkpoint_path = f"/home/harry/Documents/driving_models/{name}.cpt"
    callbacks = [
        ModelIntervalCheckpoint(checkpoint_path, interval=10000),
        FileLogger(log_path, interval=1000),
    ]
    history = dqn.fit(
        env,
        nb_steps=TRAIN_STEPS,
        visualize=False,
        verbose=2,
        action_repetition=action_rep,
        nb_max_episode_steps=None,
        callbacks=callbacks,
    )
    dqn.save_weights(checkpoint_path, overwrite=True)


class SpeedLimitPolicy(Policy):
    def __init__(self, env, max_speed, eps=0):
        self.max_speed = max_speed
        self.env = env
        self.eps = eps

    def select_action(self, q_values):
        if np.random.uniform() < self.eps:
            return np.random.randint(0, NB_ACTIONS)
        else:
            speed = get_features_from_env(env)[0]
            order = np.argsort(q_values)
            best = order[-1]
            if speed > self.max_speed and best == 1:
                return order[-2]
            else:
                return best



# Finally, evaluate our algorithm for 5 episodes.
def test(env, dqn, name, action_rep):
    checkpoint_path = f"/home/harry/Documents/driving_models/{name}.cpt"
    dqn = load(dqn, checkpoint_path)
    dqn.test(
        env,
        nb_episodes=5,
        visualize=True,
        action_repetition=action_rep,
        # nb_max_start_steps=40,
        # start_step_policy=start_policy,
        nb_max_episode_steps=None,
    )


if __name__ == "__main__":
    command = sys.argv[1]
    name = sys.argv[2]
    params = load_json(f"params/{name}.json")
    print(params)
    if command == "train":
        env = get_driving_env_dqn(
            name, params["grass_penalty"], params["still_penalty"]
        )
        # speed_limit_policy = LinearAnnealedPolicy(
        #     SpeedLimitPolicy(env, 3000), "eps", 1, 0.1, 0.03, ANNEAL_STEPS
        # )
        # dqn = get_dqn(env, speed_limit_policy)
        dqn = get_dqn(env, annealed_policy)
        train(env, dqn, name, params["action_rep"])
    if command == "resume":
        env = get_driving_env_dqn(
            name, params["grass_penalty"], params["still_penalty"]
        )
        dqn = get_dqn(env, eps_greedy_policy)
        train(env, dqn, name, params["action_rep"])
    if command == "test":
        env = get_driving_env_dqn("test_env", 0.01, 0.01)
        dqn = get_dqn(env, eps_greedy_policy)
        test(env, dqn, name, params["action_rep"])

    # for f in os.listdir('params'):
    #     params = load_json(f'params/{f}')
    #     name = f[:-5]
    #     if name != 'baseline':
    #         env = get_driving_env_dqn(name, params['grass_penalty'], params['still_penalty'])
    #         dqn = get_dqn(env)
    #         train(env, dqn, name, params['action_rep'])
