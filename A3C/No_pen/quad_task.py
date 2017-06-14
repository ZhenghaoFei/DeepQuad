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
        if terminal and info!='time_out':
            print info
            reward = -1000
            return reward

        current_position = states[0:3]
        # last_position = state_last[0:3]

        error = np.mean((current_position - self.hover_position_set)**2)
        error = max(error, 0.01)
        # error += 1e-7 # prevent div by 0
        reward = 1/error
        # reward = min(reward, 10000)

        # reward = -np.mean((current_position - self.hover_position_set)**2)/100 

        return reward

class hover1(object):
    """hover"""
    def __init__(self, hover_position_set):
        self.hover_position_set = hover_position_set

    def reward(self, states, terminal, info):
        # this function is aimed to let the quadcopter hover in a certain position.
        # e.g
        # hover_position = np.asarray([0, 0, 0]) # pn = 0, pe = 0, pd = 0
        if terminal and info!='time_out':
            print info
            reward = -100
            return reward

        current_position = states[0:3]
        # last_position = state_last[0:3]

        # error = np.mean((current_position - self.hover_position_set)**2)
        # error += 1e-7 # prevent div by 0
        # reward = 10/error
        # reward = min(reward, 10000)

        reward = -np.mean((current_position - self.hover_position_set)**2)/100 

        return reward

class hover2(object):
    """hover"""
    def __init__(self, hover_position_set):
        self.hover_position_set = hover_position_set

    def reward(self, states, terminal, info):
        # this function is aimed to let the quadcopter hover in a certain position.
        # e.g
        # hover_position = np.asarray([0, 0, 0]) # pn = 0, pe = 0, pd = 0
        if terminal and info!='time_out':
            print info
            reward = -100
            return reward
        else:
            reward = 10
            return reward

        # current_position = states[0:3]
        # last_position = state_last[0:3]

        # error = np.mean((current_position - self.hover_position_set)**2)
        # error += 1e-7 # prevent div by 0
        # reward = 10/error
        # reward = min(reward, 10000)

        # reward = -np.mean((current_position - self.hover_position_set)**2)/100 

        return reward

class hover3(object):
    """hover"""
    def __init__(self, hover_position_set):
        self.hover_position_set = hover_position_set

    def reward(self, states, terminal, info):
        # this function is aimed to let the quadcopter hover in a certain position.
        # e.g
        # hover_position = np.asarray([0, 0, 0]) # pn = 0, pe = 0, pd = 0
        if terminal and info!='time_out':
            # print info
            reward = -1000
            return reward
        else:
            reward = 1
            return reward

        # current_position = states[0:3]
        # last_position = state_last[0:3]

        # error = np.mean((current_position - self.hover_position_set)**2)
        # error += 1e-7 # prevent div by 0
        # reward = 10/error
        # reward = min(reward, 10000)

        # reward = -np.mean((current_position - self.hover_position_set)**2)/100 

        return reward

class hover4(object):
    """hover"""
    def __init__(self, hover_position_set):
        self.hover_position_set = hover_position_set

    def reward(self, states, terminal, info):
        # this function is aimed to let the quadcopter hover in a certain position.
        # e.g
        # hover_position = np.asarray([0, 0, 0]) # pn = 0, pe = 0, pd = 0
        if terminal and info!='time_out':
            # print info
            reward = -100
            return reward
        else:
            reward = 100
            return reward

        # current_position = states[0:3]
        # last_position = state_last[0:3]

        # error = np.mean((current_position - self.hover_position_set)**2)
        # error += 1e-7 # prevent div by 0
        # reward = 10/error
        # reward = min(reward, 10000)

        # reward = -np.mean((current_position - self.hover_position_set)**2)/100 

        return reward