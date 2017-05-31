#!/usr/bin/python2.7
# Filename: simulator.py
# Description: This file run and test the simulator and plot state figures
# Auther: Zhenghao Fei,  Peng Wei

import numpy as np
import matplotlib.pyplot as plt
from simulator import QuadCopter


def plot_states(states):
    # plot
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 10), sharey=True)
    axes[0, 0].plot(states[:,0])
    axes[0, 0].set_title('pn')
    axes[0, 1].plot(states[:,1])
    axes[0, 1].set_title('pe')
    axes[0, 2].plot(-states[:,2])
    axes[0, 2].set_title('pd')
    axes[1, 0].plot(states[:,3])
    axes[1, 0].set_title('u')
    axes[1, 1].plot(states[:,4])
    axes[1, 1].set_title('v')
    axes[1, 2].plot(states[:,5])
    axes[1, 2].set_title('w')
    axes[2, 0].plot(states[:,6])
    axes[2, 0].set_title('phi')
    axes[2, 1].plot(states[:,7])
    axes[2, 1].set_title('theta')
    axes[2, 2].plot(states[:,8])
    axes[2, 2].set_title('psi')
    axes[3, 0].plot(states[:,9])
    axes[3, 0].set_title('p')
    axes[3, 1].plot(states[:,10])
    axes[3, 1].set_title('q')
    axes[3, 2].plot(states[:,11])
    axes[3, 2].set_title('r')
    axes[4, 0].plot(states[:,12])
    axes[4, 0].set_title('pen_x')
    axes[4, 1].plot(states[:,13])
    axes[4, 1].set_title('pen_y')
    axes[4, 2].plot(states[:,14])
    fig.subplots_adjust(hspace=1.4) 
    plt.show()

def main():
    quad  = QuadCopter()
    time  = 10 # sec
    steps = int(time/quad.Ts)
    delta    = [1.5, 1.5, 1.5, 1.5]

    print "Simulate %i sec need total %i steps" %(time, steps)

    states = np.zeros([steps, quad.stateSpace])
    for i in range(steps):
        state = quad.step(delta)
        states[i] = state

    plot_states(states)


if __name__ == "__main__":
    main()