import numpy as np


class HandCrafted:
    def __init__(self):
        pass

    def store_data(self, state, state_prime, reward, actions_prime, done):
        pass

    @staticmethod
    def choose_action(actions):
        i = np.argmax([-4 * i[2] - i[4] - i[5] - i[1] - i[3] + i[6] for i in actions])
        return i

    def learn(self):
        pass
