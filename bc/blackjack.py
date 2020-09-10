import random
from collections import defaultdict

import gym
import plotly.express as px
from tqdm import tqdm

from common.utils import load_pkl, print_step, save_pkl

env = gym.make("Blackjack-v0")


def run_blackjack(policy, show=False):
    env = gym.make("Blackjack-v0")
    obs = env.reset()
    history = []
    while True:
        move = policy(obs)
        new_obs, reward, done, info = env.step(move)
        history.append((obs, move, new_obs, done, reward))
        obs = new_obs
        if done:
            break
    if show:
        result = history[-1][-1]
        if result > 0:
            print("Congrats! You win")
        if result == 0:
            print("Tie!")
        if result < 0:
            print("You lose!")
        print("Your cards: ", env.player)
        print("Dealers cards", env.dealer)

    env.close()
    return history


def average_reward(env, policy, trials=10):
    rewards = 0
    for _ in range(trials):
        rewards += run_blackjack(env, policy)[-1][-1]
    return rewards / trials


def threshold_policy(obs, thresh):
    if obs[0] < thresh:
        return 1
    else:
        return 0


def threshold_generator(thresh):
    return lambda obs: threshold_policy(obs, thresh)


def random_policy(obs):
    return random.choice([0, 1])


def human_blackjack_player(obs):
    print(f"Your total: {obs[0]}")
    print(f"Dealer card: {obs[1]}")
    print(f"Usable ace?: {obs[2]}")
    return int(input("0 for stay 1 for hit: "))


def compare_thresholds(env):
    averages = [
        average_reward(env, threshold_generator(thresh), 10000) for thresh in range(20)
    ]
    f = px.line(averages)
    f.show()


def q_policy(obs, q):
    return 0 if q[(obs, 0)] >= q[(obs, 1)] else 1


def eps_policy(obs, q, eps):
    explore = random.random() <= eps
    if explore:
        return random.choice([0, 1])
    else:
        return 0 if q[(obs, 0)] >= q[(obs, 1)] else 1


def q_learn(env, eps, gamma, alpha, epochs):
    q_table = defaultdict(int)
    training_results = []
    for _ in tqdm(range(epochs)):
        history = run_blackjack(env, lambda obs: eps_policy(obs, q_table, eps))
        for s, a, s_next, done, r in reversed(history):
            old = q_table[(s, a)]
            if done:
                q_table[(s, a)] = (1 - alpha) * old + alpha * (r)
            else:
                best_next = max([q_table[(s_next, 0)], q_table[(s_next, 1)]])
                q_table[(s, a)] = (1 - alpha) * old + alpha * (r + gamma * best_next)
        training_results.append(history[-1][-1])
    return q_table, training_results


eps_policy_generator = lambda table, eps: lambda obs: eps_policy(obs, table, eps)


def train_qlearn(env, eps, gamma, alpha, epochs):
    table, training_history = q_learn(env, eps, gamma, alpha, epochs)
    save_pkl(table, "model.pkl")


# EPS = 0.4
# GAMMA = 1
# ALPHA = 0.01
# EPOCHS = 1000000
# train_qlearn(EPS, GAMMA, ALPHA, EPOCHS)

# df = pd.DataFrame()
# df['wins'] = training_history
# px.line(df.wins.rolling(window=10).mean()).show()
# save_pkl(table, 'model.pkl')

# table = load_pkl("model.pkl")
# print(table)

# q_learn_player = lambda obs: eps_policy(obs, table, 0)

# print(average_reward(env, q_learn_player, 100000))
