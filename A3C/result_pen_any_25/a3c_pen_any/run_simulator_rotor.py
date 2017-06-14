#!/usr/bin/python2.7
# Filename: simulator.py
# Description: Direct input rather than RPM,
# This file run and test the simulator and plot state figures
# Auther: Zhenghao Fei,  Peng Wei

import numpy as np
import matplotlib.pyplot as plt
from simulator_rotor_input import QuadCopter
from util import *

SIM_TIME_STEP = 0.1

def main():
    quad  = QuadCopter(Ts = SIM_TIME_STEP, inverted_pendulum=True)
    time  = 10.0 # sec
    steps = int(time/quad.Ts)
    f = 1
    rotors  = [0.5, 0.5, 0.5, 0.5]

    print "Simulate %i sec need total %i steps" %(time, steps)

    states = np.zeros([steps, quad.stateSpace])
    for i in range(steps):
        # kk = np.asarray(uu) + np.random.rand()*0.1
        state ,terminated ,info = quad.step(rotors, continous_input=True, naive_int=True)
        if terminated:
            print info
            quad.reset()
        print state[0:3]
        
        states[i] = state
    plot_states(states)


if __name__ == "__main__":
    main()