# basic Q learning agent
from collections import defaultdict
import numpy as np
import random


class QLearning:
    def __init__(self, num_features, actions=None):
        self.num_features = num_features
        self.actions = actions
        self.alpha = .1
        self.gamma = .95
        self.epsilon = .15
        self.qq = defaultdict(lambda: np.zeros(4))  # 4 actions

    def choose_action(self, state, actions=None):
        """
        choose the action using epsilon greedy
        :param state: current state
        :param actions: use this if the actions change per state. If the actions are always the same it's cleaner to
        init actions when making class
        :return: action selected by epsilon greedy
        """
        if actions is None:
            actions = self.actions
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(actions)
        else:
            action = self.argmax(self.qq[state])
        return action

    @staticmethod
    def argmax(x):
        """
        argmax that breaks ties randomly
        :param x: anything that can be turned into an NP array
        :return:
        """
        x = np.array(x)
        return np.random.choice(np.flatnonzero(x == x.max()))

    def learn(self, state, action, reward, state_):
        """
        implementation of a single Q learning update
        :param state:
        :param action:
        :param reward:
        :param state_: state prime
        """
        max_qq_s_ = np.max(self.qq[state_])
        qq_s_a = self.qq[state][action]
        update = self.alpha * (reward + self.gamma * max_qq_s_ - qq_s_a)
        self.qq[state][action] += update
