from common.driving_utils import human_race
import gym

ROAD_COLORS = [(102 + i, 102 + i, 102 + i) for i in range(6)]

def is_still(rew):
    return rew < 0

class PenalizeLeavingRoad(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)


    def reward(self, rew):
        left_col = tuple(self.env.state[65, 50, :])
        right_col = tuple(self.env.state[65, 45, :])
        if left_col not in ROAD_COLORS or right_col not in ROAD_COLORS:
            return -50
        return rew

class PenalizeNotMoving(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, rew):
        left_col = tuple(self.env.state[65, 50, :])
        right_col = tuple(self.env.state[65, 45, :])
        if left_col not in ROAD_COLORS or right_col not in ROAD_COLORS:
            return -50
        if is_still(rew):
            return -0.5
        return rew

if __name__ == "__main__":

    human_race(PenalizeNotMoving)