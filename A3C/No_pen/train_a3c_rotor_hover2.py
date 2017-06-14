
import threading
import multiprocessing
#import matplotlib.pyplot as plt

import scipy.signal
# get_ipython().magic(u'matplotlib inline')
# from helper import *
# from vizdoom import *
import os
from simulator_rotor_input import *
# from  util import *
from quad_task import *

from random import choice
from time import sleep
from time import time
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

# ==========================
#   Training Parameters
# =========================
# Simulation step
SIM_TIME_STEP = 0.1
SAVE_STEP = 100
LEARNING_RATE = 1e-5
# Max episode length
MAX_EP_TIME = 5 # second
MAX_EP_STEPS = int(MAX_EP_TIME/SIM_TIME_STEP)


max_episode_length = MAX_EP_STEPS
gamma = .99 # discount rate for advantage estimation and reward discounting
load_model = False
model_path = './result_hover_nopen/model'



# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def sample_gaussian(mean, std):
    """
    sample from gaussian
    """
    sample = tf.random_normal([1], mean=mean, stddev=std, dtype=tf.float32, seed=None, name=None)

    return sample

# ### Actor-Critic Network


class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            # self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            # self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
            #     inputs=self.imageIn,num_outputs=16,
            #     kernel_size=[8,8],stride=[4,4],padding='VALID')
            # self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
            #     inputs=self.conv1,num_outputs=32,
            #     kernel_size=[4,4],stride=[2,2],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.inputs),256,activation_fn=tf.nn.elu)
            # hidden = tf.layers.batch_normalization(hidden)

            # hidden = slim.fully_connected(hidden, 128,activation_fn=tf.nn.elu)
            # hidden = tf.layers.batch_normalization(hidden)

            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.inputs)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            # rnn_out = tf.layers.batch_normalization(rnn_out)

            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            
            # self.policy = tf.nn.softmax(tf.clip_by_value(self.policy ,1e-10,1.0))
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy ,1e-10,1.0)))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

# ### Worker Agent



class Worker():
    def __init__(self,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.a_size = a_size
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("./result_hover_nopen/train_"+str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
        
        self.actions = self.actions = np.identity(a_size,dtype=bool).tolist()
        #End Doom set-up
        self.env = QuadCopter(SIM_TIME_STEP, max_time = MAX_EP_TIME, inverted_pendulum=False)
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:rnn_state[0],
            self.local_AC.state_in[1]:rnn_state[1]}
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
        
    def work(self, task, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                # print "episode_count ", episode_count
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_states = []
                episode_reward = 0
                episode_step_count = 0
                terminal = False
                
                
                s = self.env.reset()
                episode_states.append(s)
                rnn_state = self.local_AC.state_init
                # print "terminate: ", self.env.terminated
                while self.env.terminated == False:

                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                        feed_dict={self.local_AC.inputs:[s],
                        self.local_AC.state_in[0]:rnn_state[0],
                        self.local_AC.state_in[1]:rnn_state[1]})

                    # print "a_dist[]: ", a_dist[0]
                    a_dist[0] = np.abs(a_dist[0])
                    a_dist[0] = np.clip(a_dist[0], 1e-10, 1.0)
                    a = np.random.choice(self.a_size, p=a_dist[0])
                    
                    # a = np.argmax(a_dist == a)

                    s2, terminal, info = self.env.step(a)
                    # print s2
                    # print "a: ", a
                    # print "self.actions[a]: ", self.actions[a]
                    # print "terminal: ", terminal
                    r = task.reward(s2, terminal, info) # calculate reward basec on s2
                    # print r
                    if terminal == False:
                        s1 = s2
                        episode_states.append(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s,a,r,s1,terminal,v[0,0]])
                    episode_values.append(v[0,0])
                    last_state = np.copy(s)
                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and terminal != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if terminal == True:
                        break
                                            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
                                
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % SAVE_STEP == 0 and episode_count != 0:

                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")
                        
                    print "episode_count: ", episode_count
                    print "last state: ", last_state[0:3]
                    mean_reward = np.mean(self.episode_rewards[-SAVE_STEP:])
                    mean_length = np.mean(self.episode_lengths[-SAVE_STEP:])
                    mean_value = np.mean(self.episode_mean_values[-SAVE_STEP:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                    print "mean_reward: ", mean_reward
                    # print "mean_length: ", mean_length
                    print "mean_value: ", mean_value

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1




env  = QuadCopter(SIM_TIME_STEP, max_time = MAX_EP_TIME, inverted_pendulum=False)

state_dim = env.stateSpace
action_dim = env.actionSpace
action_limit = env.actionLimit
hover_position = np.asarray([0, 0, 0])
task = hover(hover_position)

print("Quadcopter created")
print('state_dim: ', state_dim)
print('action_dim: ', action_dim)
print('action_limit: ',action_limit)
print('max time: ', MAX_EP_TIME)
print('max step: ',MAX_EP_STEPS)     
print("hover_position: ", hover_position)
print("learning rate: ", LEARNING_RATE)

num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
num_workers = 8

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    master_network = AC_Network(state_dim,action_dim,'global',None) # Generate global network

    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(i,state_dim,action_dim,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []


    for worker in workers:
        worker_work = lambda: worker.work(task, max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)

