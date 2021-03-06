# # This file contains utilities such as plot
import matplotlib.pyplot as plt
import numpy as np

def plot_states(states):
    # plot
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 10), sharey=True)
    axes[0, 0].plot(states[:,0])
    axes[0, 0].set_title('pn')
    axes[0, 1].plot(states[:,1])
    axes[0, 1].set_title('pe')
    axes[0, 2].plot(-states[:,2])
    axes[0, 2].set_title('h')
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
    axes[4, 2].set_title('pen_vx')
    fig.subplots_adjust(hspace=1.4) 
    # plt.show()

