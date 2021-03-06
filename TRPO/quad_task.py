# This file defines multiple tasks for quad
import numpy as np

class hover(object):
    """hover"""
    def __init__(self, hover_position_set):
        self.hover_position_set = hover_position_set

    def reward(self, states, terminal, info):
        # this function is aimed to let the quadcopter hover in a certain position.
        # e.g
        # hover_position = np.asarray([0, 0, 0]) # pn = 0, pe = 0, pd = 0
        if terminal and info=='outrange':
            print info
            reward = -1000
            return reward

        if terminal and info=="pen_zeta<0":
            print info
            reward = -1000
            return reward

        current_position = states[0:3]
        error = current_position - self.hover_position_set
        # print error
        # last_position = state_last[0:3]
        reward = (-np.mean(error**2))/1000
        # print reward
        return reward
