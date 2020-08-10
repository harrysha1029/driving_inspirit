import os
import random
from collections import Counter
from pathlib import Path

import gym
import numpy as np
from src.consts import ACTION2INDEX, INDEX2ACTIONS, N_ACTIONS
from src.driving_utils import get_features_from_env, get_racing_env_for_human
from src.utils import load_json, load_pkl, save_json, save_pkl


def human_generate_data(record_video=False):
    a, env = get_racing_env_for_human(record_video)
    isopen = True
    data_num = len(os.listdir("data/bc/"))
    while isopen:
        s = env.reset()
        total_reward = 0.0
        steps = 0
        imgs, features, actions = [], [], []
        while True:
            if tuple(a) in ACTION2INDEX:
                imgs.append(s)
                features.append(get_features_from_env(env))
                actions.append(ACTION2INDEX[tuple(a)])
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if steps==1500:
                path = f"data/bc/data_{data_num}.pkl"
                print(f"Saving data to {path}")
                save_pkl([imgs, features, actions], path)
                data_num += 1
                break
            if not isopen:
                break
    env.close()


def gather_data_balanced():
    base = "data/bc/"
    paths = [Path(base) / f for f in os.listdir(base) if "gathered" not in f]
    gathered_imgs = []
    gathered_features = []
    gathered_labels = []
    for path in paths:
        imgs, features, labels = load_pkl(path)
        n = min(Counter(labels).values())
        for a in range(N_ACTIONS):
            indices = random.sample([i for i, x in enumerate(labels) if x == a], n)
            gathered_imgs.extend([imgs[i] for i in indices])
            gathered_features.extend([features[i] for i in indices])
            gathered_labels.extend([labels[i] for i in indices])
    save_pkl(
        [gathered_imgs, gathered_features, gathered_labels], "data/bc/gathered.pkl"
    )

def gather_data_all():
    base = "data/bc/"
    paths = [Path(base) / f for f in os.listdir(base) if "gathered" not in f]
    gathered_imgs = []
    gathered_features = []
    gathered_labels = []
    for path in paths:
        imgs, features, labels = load_pkl(path)
        gathered_imgs.extend(imgs)
        gathered_features.extend(features)
        gathered_labels.extend(labels)
    save_pkl(
        [gathered_imgs, gathered_features, gathered_labels], "data/bc/gathered.pkl"
    )


def load_bc_data(truncate=None):
    imgs, features, labels = map(
        lambda x: np.array(x, dtype="float32"), load_pkl("data/bc/gathered.pkl")
    )
    print(Counter(labels))
    if truncate:
        return imgs[:truncate], features[:truncate], labels[:truncate]
    return imgs, features, labels


if __name__ == "__main__":
    # Don't run this as main but here's what you need to run.
    human_generate_data(False)
    gather_data()
