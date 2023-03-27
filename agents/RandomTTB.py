from agents.ttb_sequential import TakeTheBestSequential
import numpy as np


class RandomTTB(TakeTheBestSequential):
    def __init__(self, num_features, current_state):
        super().__init__(num_features, current_state)
        self.beta = np.random.rand(self.num_features)

    def learn(self):
        self.beta = np.random.rand(self.num_features)

    def store_data(self, *args):
        pass

    def feature_importance(self, *args, **kwargs):
        feature_importance = np.argsort(np.argsort(self.beta))
        return feature_importance
