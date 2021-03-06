# This file is aim to train the quadcopter keet at a constant position

import numpy as np
import tflearn
import time
from simulator import QuadCopter
from replay_buffer import ReplayBuffer
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt

import sys, os


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
    plt.show()


# ==========================
#   Training Parameters
# =========================
# Simulation step
SIM_TIME_STEP = 0.01
# Max training steps
MAX_EPISODES = 50000
SAVE_STEP = 1000
# Max episode length
MAX_EP_TIME = 2 # second
MAX_EP_STEPS = int(MAX_EP_TIME/SIM_TIME_STEP)
# Explore decay rate
EXPLORE_INIT = 1
EXPLORE_DECAY = 0.99
EXPLORE_MIN = 0.1

# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 1e-4
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 1e-3
# Discount factor 
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================

# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 1e6
MINIBATCH_SIZE = 64

# ===========================
#   Actor and Critic DNNs
# ===========================
class ActorNetwork(object):
    """ 
    Input to the network is the state, output is the action
    under a deterministic policy.

    """
    def __init__(self, sess, state_dim, action_dim, action_limit, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_limit = action_limit
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        
        # Combine the gradients here 
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self): 
        # inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim], name='state')
        # net = tf.layers.batch_normalization(inputs)
        # net = layers.fully_connected(inputs, num_outputs=1024, weights_initializer=layers.xavier_initializer() ,activation_fn=tf.nn.relu)
        # net = tf.layers.batch_normalization(net)
        # # net = tf.expand_dims(net, 1)
        # # net = tf.expand_dims(net, -1)
        # # net = layers.conv2d(net, num_outputs=32, kernel_size=1, stride=1,padding='SAME', activation_fn=tf.nn.relu)
        # # net = tf.layers.batch_normalization(net)
        # # net = layers.conv2d(net, num_outputs=2, kernel_size=1, stride=1,padding='SAME', activation_fn=tf.nn.relu)
        # # net = tf.layers.batch_normalization(net)
        # # net = layers.flatten(net)
        # net = layers.fully_connected(net, num_outputs=512, weights_initializer=layers.xavier_initializer() ,activation_fn=tf.nn.relu)
        # net = tf.layers.batch_normalization(net)
        # # net = layers.fully_connected(net, num_outputs=512, weights_initializer=layers.xavier_initializer() ,activation_fn=tf.th)
        # out_w = tf.Variable(np.random.randn(512, self.a_dim)*3e-3, dtype=tf.float32, name="out_w")
        # out_b = tf.Variable(tf.zeros([self.a_dim]), dtype=tf.float32, name="out_b")
        # out = tf.tanh(tf.matmul(net, out_w) +out_b)
        # # out = layers.fully_connected(net, num_outputs=self.a_dim, weights_initializer=layers.xavier_initializer() ,activation_fn=tf.nn.sigmoid)
        # scaled_out = tf.multiply(out, self.action_limit)# Scale output to -action_limit to action_limit

        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')
        net = tflearn.fully_connected(net, 300, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_limit)

        return inputs, out, scaled_out

        return inputs, out, scaled_out 

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

class CriticNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]
    
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        # self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        # inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim])
        # action = tf.placeholder(dtype=tf.float32, shape=[None, self.a_dim])

        # net = layers.fully_connected(inputs, num_outputs=1024, weights_initializer=layers.xavier_initializer() ,activation_fn=tf.nn.relu)
        # # # Add the action tensor in the 2nd hidden layer
        # # # Use two temp layers to get the corresponding weights and biases 
        # # net = tf.layers.batch_normalization(net)
        # # net = tf.expand_dims(net, 1)
        # # net = tf.expand_dims(net, -1)
        # # net = layers.conv2d(net, num_outputs=32, kernel_size=1, stride=1,padding='SAME', activation_fn=tf.nn.relu)
        # # # net = tf.layers.batch_normalization(net)
        # # net = layers.conv2d(net, num_outputs=2, kernel_size=1, stride=1,padding='SAME', activation_fn=tf.nn.relu)
        # # # net = tf.layers.batch_normalization(net)
        # # net = layers.flatten(net)
        # t1 = layers.fully_connected(net, num_outputs=512, weights_initializer=layers.xavier_initializer(), activation_fn=None)
        # t2 = layers.fully_connected(action, num_outputs=512, weights_initializer=layers.xavier_initializer(), activation_fn=None)

        # net = tf.nn.relu(t1 + t2)
        # net = tf.layers.batch_normalization(net)
        # out_w = tf.Variable(np.random.randn(512, 1)*3e-3, dtype=tf.float32)
        # out_b = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="out_b")

        # out = tf.matmul(net, out_w) + out_b

        # out = layers.fully_connected(net, num_outputs=1, weights_initializer=layers.xavier_initializer(), activation_fn=None)

        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions): 
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
    success_rate = tf.Variable(0.)
    tf.summary.scalar("Success Rate", success_rate)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [success_rate, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("total_parameters:", total_parameters)

# ===========================
#   Reward functions
# ===========================
def reward_function_hover_decorator(hover_position_set):

    def reward_function_hover(states, terminal, info):
        # this function is aimed to let the quadcopter hover in a certain position.
        # e.g
        # hover_position = np.asarray([0, 0, 0]) # pn = 0, pe = 0, pd = 0

        if terminal:
            reward = -500
            print "terminated " , info
        else:
            current_position = states[0:3]
            # reward function = -MSE(current_position, hover_position)
            reward = -np.mean((current_position - hover_position_set)**2) + 200
            # print reward

        return reward

    return reward_function_hover
# ===========================
#   Agent Training
# ===========================


def train(sess, env, actor, critic, reward_fc):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # load model if have
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(SUMMARY_DIR)
    
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        print("global step: ", global_step.eval())

    else:
        print ("Could not find old network weights")

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()
    count_parameters()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
    tic = time.time()
    last_epreward = 0 
    explore = EXPLORE_INIT

    for i in xrange(MAX_EPISODES):
        explore *= EXPLORE_DECAY
        explore = max(EXPLORE_MIN, explore)
        print ("explore: ", explore)
        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        states = np.zeros([MAX_EP_STEPS, env.stateSpace])

        if i % SAVE_STEP == 0 and i!= 0: # save check point every xx episode
            save_path = saver.save(sess, SUMMARY_DIR + "model.ckpt" , global_step = i)
            print("Model saved in file: %s" % save_path)

        for j in xrange(MAX_EP_STEPS):

            # Added exploration noise
            exp = np.random.rand() * explore * env.actionLimit
            a = actor.predict(np.reshape(s, (1, 16))) + exp
            # a = actor.predict(np.reshape(s, (1, 16))) + (1. / (1. + i))

            s2, terminal, info = env.step(a[0])
            states[j] = s2

            r = reward_fc(s2, terminal, info) # calculate reward basec on s2

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r, \
                terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:     
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in xrange(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
            
                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)                
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal or j == MAX_EP_STEPS-1 or r < -10000:
                # if i > 30:
                #     plot_states(states)

                print s[0:3]
                time_gap = time.time() - tic

                # summary_str = sess.run(summary_ops, feed_dict={
                #     summary_vars[0]: ep_reward,
                #     summary_vars[1]: ep_ave_max_q / float(j)
                # })

                # writer.add_summary(summary_str, i)
                # writer.flush()
                # if ep_reward < last_epreward and last_epreward != 0:
                #     actor.learning_rate /= 10
                #     critic.learning_rate /= 10
                #     print "lr decay to ", actor.learning_rate
                last_epreward = ep_reward
                print '| Reward: %.2f' % int(ep_reward/(j+1)), " | Episode", i, \
                        '| Qmax: %.4f' % (ep_ave_max_q / float(j+1)), ' | Time: %.2f' %(time_gap)
                tic = time.time()

                break


def main(_):
        
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env  = QuadCopter(SIM_TIME_STEP, inverted_pendulum=False)

        state_dim = env.stateSpace
        action_dim = env.actionSpace
        action_limit = env.actionLimit

        print("Quadcopter created")
        print('state_dim: ', state_dim)
        print('action_dim: ', action_dim)
        print('action_limit: ',action_limit)
        print('max time: ', MAX_EP_TIME)
        print('max step: ',MAX_EP_STEPS)        

        hover_position = np.asarray([0, 0, -50])
        reward_fc = reward_function_hover_decorator(hover_position)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            actor = ActorNetwork(sess, state_dim, action_dim, action_limit, \
                ACTOR_LEARNING_RATE, TAU)

            critic = CriticNetwork(sess, state_dim, action_dim, \
                CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

            train(sess, env, actor, critic, reward_fc)

if __name__ == '__main__':
    tf.app.run()
