# # This file contains utilities such as plot
import matplotlib.pyplot as plt
import numpy as np

def plot_states(states):
    # plot
    # fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 10), sharey=True)
    states_label = ['pn', 'pe', '-h', 'u', 'v', 'w', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'pen_x', 'pen_y', 'pen_vx', 'pen_vy', 'in1', 'in2', 'in3', 'in4']
    
    # 'pn', 'pe', 'h', 'u', 'v', 'w'
    fig = plt.figure(1)
    for i in range(6):
        ax = plt.subplot(2,3,i+1)
        ax.plot(states[:, i])
        ax.set_title(states_label[i])
        fig.subplots_adjust(hspace=0.4)
    
    # 'phi', 'theta', 'psi', 'p', 'q', 'r'
    fig = plt.figure(2)
    for i in range(6):
        ax = plt.subplot(2,3,i+1)
        ax.plot(states[:, i+6])
        ax.set_title(states_label[i+6])
        fig.subplots_adjust(hspace=0.4)    


    # 'in1', 'in2', 'in3', 'in4'
    fig = plt.figure(3)
    for i in range(4):
        ax = plt.subplot(2,2,i+1)
        ax.plot(states[:, i+16])
        ax.set_title(states_label[i+16])
        fig.subplots_adjust(hspace=0.4) 

    # 'pen_x', 'pen_y'
    fig = plt.figure(4)
    for i in range(2):
        ax = plt.subplot(1,2,i+1)
        ax.plot(states[:, i+12])
        ax.set_title(states_label[i+12])
        fig.subplots_adjust(hspace=0.4)  

    plt.show()

def save_states(states, Ts):
    filename = "./state.csv"
    length = states.shape[0]
    time = np.array([Ts * i for i in range(length)])
    time = np.expand_dims(time, axis=1)
    states = np.concatenate((states, time), axis=1)
    np.savetxt(filename, states, delimiter=',')
    print "states saved at: ", filename
    print "states shape: ", states.shape














