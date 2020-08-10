import json
import pickle


def print_step(obs, reward, done, info):
    print(f"{obs=}\n{reward=}\n{done=}\n{info=}")


def save_pkl(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def save_json(obj, fname):
    with open(fname, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(fname):
    with open(fname, "r") as f:
        return json.load(f)
