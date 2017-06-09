
import numpy as np
import matplotlib.pyplot as plt
import time
from simulator_dis import QuadCopter
from util import *
from quad_task import *
from replay_buffer import ReplayBuffer

# import tflearn
import tensorflow as tf
import tensorflow.contrib.layers as layers

PLOT = True

# ==========================
#   Training Parameters
# ==========================

# Simulation step
SIM_TIME_STEP = 0.1
# Max training steps
MAX_EPISODES = 500000
# Max episode length
MAX_EP_TIME = 5 # second
MAX_EP_STEPS = int(MAX_EP_TIME/SIM_TIME_STEP)

# # Explore decay rate
# EXPLORE_INIT = 0.5
# EXPLORE_DECAY = 0.999
# EXPLORE_MIN = 0.01

# Base learning rate for the Qnet Network
Q_LEARNING_RATE = 1e-4
# Discount factor 
GAMMA = 0.9

# Soft target update param
TAU = 0.001
TARGET_UPDATE_STEP = 100

MINIBATCH_SIZE = 64
SAVE_STEP = 100
EPS_MIN = 0.05
EPS_DECAY_RATE = 0.9999
# ===========================
#   Utility Parameters
# ===========================

# Directory for storing tensorboard summary results
SUMMARY_DIR = './results_dqn_move/'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 1000000
EVAL_EPISODES = 1


# ===========================
#   Q DNN
# ===========================
class QNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the Qnet network
        self.inputs, self.out = self.create_Q_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out = self.create_Q_network()
        
        self.target_network_params = tf.trainable_variables()[(len(self.network_params)):]


        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. -self.tau))
                for i in range(len(self.target_network_params))]
    
        # Network target (y_i)
        self.observed_q_value = tf.placeholder(tf.float32, [None])
        self.action_taken = tf.placeholder(tf.float32, [None, self.a_dim])
        self.predicted_q_value = tf.reduce_sum(tf.multiply(self.out, self.action_taken), reduction_indices = 1) 

        # Define loss and optimization Op
        self.Qnet_global_step = tf.Variable(0, name='Qnet_global_step', trainable=False)

        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.observed_q_value))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.Qnet_global_step)


    def create_Q_network(self):
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim])

        net = layers.fully_connected(inputs, num_outputs=512 ,activation_fn=tf.nn.relu)
        net = tf.layers.batch_normalization(net)

        # net = layers.fully_connected(net, num_outputs=256 ,activation_fn=tf.nn.relu)
        # net = tf.layers.batch_normalization(net)

        net = layers.fully_connected(net, num_outputs=128 ,activation_fn=tf.nn.relu)
        net = tf.layers.batch_normalization(net)

        out = layers.fully_connected(net, num_outputs=self.a_dim ,activation_fn=None)

        return inputs, out

    def train(self, inputs, action, observed_q_value):

        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action_taken: action,
            self.observed_q_value: observed_q_value
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
    reward = tf.Variable(0.)
    tf.summary.scalar('Rewards', reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar('Qmax Value', episode_ave_max_q)

    summary_vars = [reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, task, Qnet, global_step):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())

    # load model if have
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(SUMMARY_DIR)
    
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        print("global step: ", global_step.eval())

    else:
        print ("Could not find old network weights")

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    Qnet.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    i = global_step.eval()


    eval_acc_reward = 0
    tic = time.time()
    eps = 1

    # setup plot
    if PLOT:
        plt.ion()
        fig1 = plt.figure()
        plot1 = fig1.add_subplot(131)
        plot2 = fig1.add_subplot(132)
        plot3 = fig1.add_subplot(133)

    while True:
        i += 1
        eps = EPS_DECAY_RATE**i
        eps = max(eps, EPS_MIN)
        s = env.reset()
        # plt.imshow(s, interpolation='none')
        # plt.show()
        # s = prepro(s)
        ep_ave_max_q = 0
        states = []
        if i % SAVE_STEP == 0 : # save check point every 1000 episode
            sess.run(global_step.assign(i))
            save_path = saver.save(sess, SUMMARY_DIR + "model.ckpt" , global_step = global_step)
            print("Model saved in file: %s" % save_path)
            print("Successfully saved global step: ", global_step.eval())


        for j in xrange(MAX_EP_STEPS+1):
            predicted_q_value = Qnet.predict(np.reshape(s, np.hstack((1, Qnet.s_dim))))
            predicted_q_value = predicted_q_value[0]

            np.random.seed()

            action = np.argmax(predicted_q_value)
            # if np.random.rand() < eps:
            #     action = np.random.randint(env.actionSpace)
                # print('eps')
            # print'actionprob:', action_prob

            # print(action)
            # print(a)
            states.append(s)
            s2, terminal, info = env.step(action)
            r = task.reward(s2, terminal, info) # calculate reward basec on s2


            eval_acc_reward += r

            if terminal:

              # plot
                if PLOT:
                    states = np.asarray(states)
                    plot1.plot(states[:,0])    
                    plot2.plot(states[:,1])  
                    plot3.plot(states[:,2])  
                    plt.pause(0.001)
                # summary

                time_gap = time.time() - tic

                print s[0:3]
                print ('| Reward: %i ' % (eval_acc_reward/float(EVAL_EPISODES)), "| Episode", i, \
                    '| Qmax: %.4f' % (ep_ave_max_q / float(j+1)), ' | Time: %.2f' %(time_gap), ' | Eps: %.2f' %(eps))
                tic = time.time()

                # print(' 100 round reward: ', eval_acc_reward)
                eval_acc_reward = 0

                break

            s = s2


def prepro(state):
    """ prepro state to 3D tensor   """
    # print('before: ', state.shape)
    state = state.reshape(state.shape[0], state.shape[1], 1)
    # print('after: ', state.shape)
    # plt.imshow(state, interpolation='none')
    # plt.show()
    # state = state.astype(np.float).ravel()
    return state

def action_ecoder(action, action_dim):
    action_vector = np.zeros(action_dim, dtype=np.float32)
    action_vector[action] = 1
    return action_vector


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
 
        global_step = tf.Variable(0, name='global_step', trainable=False)

        env  = QuadCopter(SIM_TIME_STEP, max_time = MAX_EP_TIME, inverted_pendulum=False)
        hover_position = np.asarray([0, 10, 10])
        task = hover(hover_position)

        state_dim = env.stateSpace
        action_dim = env.actionSpace
        action_limit = env.actionLimit        
        # np.random.seed(RANDOM_SEED)
        # tf.set_random_seed(RANDOM_SEED)

        print("Quadcopter created")
        print('state_dim: ', state_dim)
        print('action_dim: ', action_dim)
        print('action_limit: ',action_limit)
        print('max time: ', MAX_EP_TIME)
        print('max step: ',MAX_EP_STEPS)   

        Qnet = QNetwork(sess, state_dim, action_dim, \
            Q_LEARNING_RATE, TAU)


        train(sess, env, task, Qnet, global_step)

if __name__ == '__main__':
    tf.app.run()
