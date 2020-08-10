import gym
import numpy as np
from gym.wrappers.monitor import Monitor
from pyglet.window import key
from src.consts import *

# def find_car(img):
#     for r in range(96):
#         for c in range(96):
#             if list(img[r, c, :]) == [204, 0, 0]:
#                 return r, c


def is_road(img, r, c):
    return all(102 <= x <= 107 for x in img[r, c, :])


def front_is_road(img):
    return all(is_road(img, r, CAR_COL) for r in range(4, CAR_ROW - 4))


def greyscale(img):
    x = np.zeros((96, 96))
    for r in range(96):
        for c in range(96):
            x[r, c] = is_road(img, r, c)
    return x


def run_racing(env, policy, show=False):
    obs = env.reset()
    # if show:
    #     print(obs)
    # history = []
    while True:
        env.render()
        move = policy(obs, env)
        new_obs, reward, done, info = env.step(move)
        obs = new_obs
        # history.append((obs, move, new_obs, done, reward))
        # if show:
        #     print_step(obs, reward, done, info)
        # if done:
        #     return history


def get_racing_env(record_video=False):
    env = gym.make("CarRacing-v0")
    env.render()
    if record_video:
        env = Monitor(env, "video", force=True)
    return env


def get_racing_env_for_human(record_video):
    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = gym.make("CarRacing-v0")
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    if record_video:
        env = Monitor(env, "video", force=True)
    return a, env


def get_features_from_env(env):
    wheel = env.car.wheels[0]
    speed = sum(map(lambda x: x ** 2, list(wheel.linearVelocity)))
    return [speed, wheel.angularVelocity] + list(wheel.linearVelocity)


def human_race():
    a, env = get_racing_env_for_human(False)
    isopen = True
    while isopen:
        s = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done:
                break
            if not isopen:
                break
    env.close()
