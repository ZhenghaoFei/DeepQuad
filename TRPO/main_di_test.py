# This version works fine for gym

import numpy as np
from simulator_di import QuadCopter
from  util import *
from quad_task import *

import tensorflow as tf
import tensorflow.contrib.layers as layers
import gym
import logz
import scipy.signal
# ==========================
#   Training Parameters
# =========================
# Simulation step
SIM_TIME_STEP = 0.01

# Max episode length
MAX_EP_TIME = 2 # second
MAX_EP_STEPS = int(MAX_EP_TIME/SIM_TIME_STEP)


def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b

def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y]. 
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1    
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the 
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)

def sample_gaussian(ac_dim, mean, std):
    """
    sample from gaussian
    """
    sample = tf.random_normal([ac_dim], mean=mean, stddev=std, dtype=tf.float32, seed=None, name=None)

    return sample


def pathlength(path):
    return len(path["reward"])

class LinearValueFunction(object):
    coef = None
    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)
    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)


def main_pendulum(logdir, seed, n_iter, gamma, min_timesteps_per_batch, initial_stepsize, desired_kl, vf_type, vf_params, animate=False):
    tf.set_random_seed(seed)
    np.random.seed(seed)

    ## Quad
    # env  = QuadCopter(SIM_TIME_STEP, inverted_pendulum=False)
    # ob_dim = env.stateSpace
    # ac_dim = 1
    # ac_lim = env.actionLimit
    # print("Quadcopter created")
    # print('state_dim: ', ob_dim)
    # print('action_dim: ', ac_dim)
    # print('action_limit: ',ac_lim)
    # print('max time: ', MAX_EP_TIME)
    # print('max step: ',MAX_EP_STEPS)        
    # hover_position = np.asarray([0, 0, -10])
    # task = hover(hover_position)

    ## Gym
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0] 
    ac_dim = env.action_space.shape[0]


    logz.configure_output_dir(logdir)
    if vf_type == 'linear':
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        vf = NnValueFunction(ob_dim=ob_dim, **vf_params)



    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None, 1], name="adv", dtype=tf.float32) # advantage function estimate
    sy_h1 = tf.nn.relu(dense(sy_ob_no, 400, "h1", weight_init=normc_initializer(1.0))) # hidden layer
    sy_h1 = tf.layers.batch_normalization(sy_h1)

    sy_h2 = tf.nn.relu(dense(sy_h1, 300, "h2", weight_init=normc_initializer(1.0))) # hidden layer
    sy_h2 = tf.layers.batch_normalization(sy_h2)

    # mean_na = dense(sy_h1, ac_dim, "mean", weight_init=normc_initializer(0.05)) # "logits", describing probability distribution of final layer
    
    # mean_na = tf.tanh(dense(sy_h2, ac_dim, "final",weight_init=normc_initializer(0.1)))*ac_lim # Mean control output
    mean_na = dense(sy_h2, ac_dim, "final",weight_init=normc_initializer(0.1)) # Mean control output
    # mean_na = tf.sigmoid(dense(sy_h2, ac_dim, "final",weight_init=normc_initializer(0.1)))*ac_lim-ac_lim/2.0 # Mean control output

    std_a =  tf.get_variable("logstdev", [ac_dim], initializer=tf.ones_initializer())


    sy_sampled_ac = sample_gaussian(ac_dim, mean_na, std_a) # sampled actions, used for defining the policy (NOT computing the policy gradient)
    # sy_prob_n = (1.0/tf.sqrt((tf.square(std_a)*2*3.1415926))) * tf.exp(-0.5*tf.square((sy_ac_n - mean_na)/std_a))
    sy_prob_n = (1.0/(std_a*2.5067)) * tf.exp(-0.5*tf.square((sy_ac_n - mean_na)/std_a))

    sy_logprob_n = tf.log(sy_prob_n)
    # sub = tf.subtract(sy_ac_n, mean_na)
    # mul = tf.multiply(sub, sy_h1)
    # sy_logprob_n = tf.log(tf.divide(sub, tf.square(std_a))) # log-prob of actions taken -- used for policy gradient calculation

    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC PURPOSES >>>>
    sy_n = tf.shape(sy_ob_no)[0]
    old_mean_na = tf.placeholder(shape=[None, ac_dim], name='old_mean_a', dtype=tf.float32) # mean_a BEFORE update (just used for KL diagnostic)
    old_std_a = tf.placeholder(shape=[ac_dim], name='old_std_a', dtype=tf.float32) # std_a BEFORE update (just used for KL diagnostic)
    # KL 
    sy_kl = tf.reduce_mean(tf.log(std_a/old_std_a) + (tf.square(old_std_a) + tf.square(old_mean_na - mean_na))/(2*tf.square(std_a)) - 0.5)
    # entropy
    sy_p_na = tf.exp(mean_na)
    sy_ent = tf.reduce_sum( - sy_p_na * mean_na) / tf.to_float(sy_n)
    # <<<<<<<<<<<<<

    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    total_timesteps = 0
    stepsize = initial_stepsize
    for i in range(n_iter):

        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps

        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            ob_last = np.copy(ob)
            terminated = False
            obs, acs, rewards = [], [], []
            j = 0
            while True:
                j += 1
                ob = ob.reshape(ob.shape[0],)
                obs.append(ob)
                # print ob
                # mean = sess.run(mean_na, feed_dict={sy_ob_no : ob[None]})[0]
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})[0]
                ## Quad
                # act = [0, 0, ac, 0, 0, 0]
                # ob, done, _ = env.step(act)
                # rew = task.reward(ob, done, _)

                ## Gym
                ob, rew, done, _ = env.step(ac)
                # ac = np.asscalar(ac)

                # ac = np.asscalar(ac)
                acs.append(ac)

                rew = np.asscalar(rew)
                rewards.append(rew)
                if done:
                    # print "done"
                    # print ob_last[0:3]
                    break       
                ob_last = np.copy(ob)
             
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            # print("return_t: ", return_t.shape)
            # print("vpred_t: ", vpred_t.shape)
            # print("adv_t: ", adv_t.shape)

            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)


        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        ac_n = ac_n.reshape([-1, ac_dim])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        standardized_adv_n = standardized_adv_n.reshape([-1, 1])

        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update
        # print standardized_adv_n
        surr, adv, logp = sess.run([sy_surr, sy_adv_n, sy_prob_n], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        _, old_mean, old_std = sess.run([update_op, mean_na, std_a], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, old_mean_na:old_mean, old_std_a:old_std})


        # KL
        if kl > desired_kl * 2: 
            stepsize /= 1.5 
            print('stepsize -> %s'%stepsize)
        elif kl < desired_kl / 2: 
            stepsize *= 1.5
            print('stepsize -> %s'%stepsize)
        else:
            print('stepsize OK')

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        # logz.log_tabular("std", old_std)
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()


def main_pendulum1(d):
    return main_pendulum(**d)

if __name__ == "__main__":

    general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=3000, initial_stepsize=1e-3)
    main_pendulum(logdir='./quad/', seed=2, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params)

    # params = [
    #     # dict(logdir='/tmp/ref/linearvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
    #     # dict(logdir='/tmp/ref/nnvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
    #     # dict(logdir='/tmp/ref/linearvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
    #     # dict(logdir='/tmp/ref/nnvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
    #     # dict(logdir='./experiment/linearvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
    #     # dict(logdir='/tmp/ref/nnvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
    # ]
    # import multiprocessing
    # p = multiprocessing.Pool()
    # p.map(main_pendulum1, params)