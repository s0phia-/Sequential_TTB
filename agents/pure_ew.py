import numpy as np


class EqualWeights:
    def __init__(self):
        pass

    def store_data(self, state, state_prime, reward, actions_prime, done):
        pass

    @staticmethod
    def choose_action(actions):
        i = np.argmax([np.sum(y) for y in actions])
        return i

    def learn(self):
        pass
