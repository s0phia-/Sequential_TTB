import numpy as np
import random


class ChasingGridWorld:
    """
    Gridworld with a monster that chases the agent with some probability, behaves randomly the rest of the time
    """
    def __init__(self, cols, rows):
        self.num_cols = cols
        self.num_rows = rows
        self.agent_position, self.monster_position = self.reset()
        self.state = self.get_state()
        self.random_monster = .1

    def reset(self):
        """
        set the agent position and monster position to random (distinct) locations on the grid
        :return: agent and monster's new positions
        """
        def random_grid_position(class_self):  # get a random grid position
            return [random.randint(class_self.num_rows), random.randint(class_self.num_cols)]

        self.agent_position = random_grid_position(self)
        self.monster_position = random_grid_position(self)

        while self.monster_position == self.agent_position:  # make sure monster and agent positions are not same
            self.monster_position = random_grid_position(self)

        return self.agent_position, self.monster_position

    def step(self, action: list):
        """
        Move the agent according to the action, then move the monster to chase the agent. Game finishes if the agent
        and mosnter are ever on the same square
        :param action: the [x, y] directions the agent should move in
        :return: the state with the agent's and monster's new position
        """
        self.agent_position += action

        if self.agent_position == self.monster_position:  # if the move takes the agent into the monster
            done = True
            reward = -100
            state = self.get_state()
            return state, reward, done, ""

        if random.uniform(0, 1) < self.random_monster:  # if the monster behaves randomly
            monster_action_x = random.randint(0, 1)
            monster_action_y = 1 - monster_action_x
            monster_action = [monster_action_x, monster_action_y]

        else:  # monster chases the agent - it moves 1 square towards the agent on the axis with the biggest difference
            diff = self.agent_position - self.monster_position
            biggest_diff_axis = np.argmax([abs(diff)])
            monster_action = [0, 0]
            monster_action[biggest_diff_axis] = diff[biggest_diff_axis] / abs(diff[biggest_diff_axis])  # gets +/- 1

        # move the monster and find updated state
        self.monster_position += monster_action
        state = self.get_state()

        # if the agent and monster share a location, the game is over
        if self.agent_position == self.monster_position:
            done = True
            reward = -100
        else:
            done = False
            reward = 1

        return state, reward, done, ""

    def render(self):
        """
        Print a grid with 0s everywhere except 1 in the agent's position and 2 in the monster's position
        """
        grid = np.zeros(self.num_rows, self.num_cols)
        grid[self.agent_position] = 1
        grid[self.monster_position] = 2
        print(grid)

    def get_state(self):
        """
        Calculate the current state. The four features are
        agent_x - monster_x
        agent_y - monster_y
        monster_x - agent_x
        monster_y - agent_y
        :return: current state
        """
        state = np.zeros(4)
        state[0:2] = self.agent_position - self.monster_position
        state[2:4] = self.monster_position - self.agent_position
        self.state = state
        return state
