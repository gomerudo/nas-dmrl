"""Use RL^2 to solve the problem of bandit with dependent arms.

In this type of problems, the experience obtained from one arm, provides
information about the other.

The original paper of Wang et al. they define the correlation of the two arms
as: p2 = 1 - p1.
"""

# from random import choice
# from time import sleep
# from time import time
# import threading
# import multiprocessing
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow.contrib.layers
# import scipy.signal

# from PIL import Image
# from PIL import ImageDraw 
# from PIL import ImageFont

import nas_rl2.rl2.helper as helper


class DependentBandit():
    """Model the dependent bandit problem."""

    def __init__(self, difficulty):
        """Construct the object.

        Params:
            difficulty (str) The level of difficulty, as specified in Wang's
            paper: uniform, easy, medium, hard. Additionally: restless.
        """
        # Number of actions in the bandit is two: two arms to be pulled
        self.num_actions = 2
        self.difficulty = difficulty

        # We reset the game at every instanciation
        self.reset()

        # Variables that will be assigned by methods
        self.bandit = None
        self.restless_list = None

    def set_restless_prob(self):
        """Set the restless probability.

        This defines the p2 = 1 - p1 correlation.
        """
        self.bandit = np.array(
            [
                self.restless_list[self.timestep],  # The arm 1: p1
                1 - self.restless_list[self.timestep]  # The arm 2: p2 = 1 - p1
            ]
        )

    def reset(self):
        """Reset the environment.

        The reset is done depending on the difficulty of the problem. This
        method will perform three main checks:
            1. If difficulty='restless': draw the restless probability function
               composed of 150 values, and build the arms with this
               distribution.
            2. If difficulty!='restless':
                a. Assign the p1 value as in Wang's
                b. Assign p2 = 1 - p1
            3. If difficulty!='independent': Draw arms with two independent
               uniformally distributed parameters p1 and p2.
        """
        self.timestep = 0

        # If difficulty is 'restless':
        #   - Define variance as a uniform random variable between 0 and 0.5
        #   - Define restless_list: the cumulative function, i.e. distribution
        #   - Standardize the restless_list: (xi - x_min)/(x_max - x_min)
        if self.difficulty == 'restless':
            variance = np.random.uniform(0, .5)

            self.restless_list = np.cumsum(
                np.random.uniform(-variance, variance, (150, 1))
            )

            self.restless_list = (
                self.restless_list - np.min(self.restless_list)
            ) / (np.max(self.restless_list - np.min(self.restless_list)))

            self.set_restless_prob()

        # If difficulty is 'easy', p1 ~ U({0.1, 0.9})
        if self.difficulty == 'easy':
            bandit_prob = np.random.choice([0.9, 0.1])

        # If difficulty is 'medium', p1 ~ U({0.25, 0.75})
        if self.difficulty == 'medium':
            bandit_prob = np.random.choice([0.75, 0.25])

        # If difficulty is 'hard', p1 ~ U({0.4, 0.6})
        if self.difficulty == 'hard':
            bandit_prob = np.random.choice([0.6, 0.4])

        # If difficulty is 'uniform', p1 ~ U({0, 1})
        if self.difficulty == 'uniform':
            bandit_prob = np.random.uniform()

        # If difficulty is not 'independent and not 'restless', set p2
        # accordingly to follow the p2 = 1 - p1.
        if self.difficulty != 'independent' and self.difficulty != 'restless':
            self.bandit = np.array([bandit_prob, 1 - bandit_prob])
        # Otherwise, if it is independent or restless, assign two independent
        # Bernoulli parameters, drawn from an uniform distribution.
        else:
            self.bandit = np.random.uniform(size=2)

    def pull_arm(self, action):
        """Perform the pulling of one arm.

        Args:
            action (int) the arm to pull on.

        Returns:
            float: The reward obtained by executing the action.
            bool: Whether or not we are done (termination flag).
            int: The timestep after performing the action.

        """
        # We make sure that, if restless, we have the density function.
        if self.difficulty == 'restless':
            self.set_restless_prob()

        # We pulled, so we make an action t=t+1
        self.timestep += 1

        # We pick one arm. Remember that each arm stores the p1 and p2
        # parameter, respectively. Hence, we obtain p_i, where i=action
        bandit = self.bandit[action]
        # We pick a random uniform value: this simulates the pull
        result = np.random.uniform()

        # If the pull is lower than the pi, it means it is under our range, and
        # we win.
        if result < bandit:
            reward = 1
        # Otherwise, we do not win anything.
        else:
            reward = 0

        # If we reached the end of the episode, we indicate it with a done flag
        if self.timestep > 99:
            done = True
        else:
            done = False

        # Return the relevant values for the action: reward, termination flag,
        # and timestep.
        return reward, done, self.timestep


class ACNet():
    """Model the Actor-Critic Network, using Tensor Flow."""

    def __init__(self, a_size, scope, trainer):
        """Construct the network.

        Params:
            a_size (int) The total number of actions allowed.
            scope (str) The scope identifier used to build the network with
                TensorFlow.
            trainer 

        """
        # variable_scope simple starts a "graph" or hierarchy in the following
        # tensorflow definitions.
        with tf.variable_scope(scope):

            # 1. Input and visual encoding layers

            # This will define the "input" (or anchor, or placeholder) for the
            # rewards that will be carried by the net.
            self.prev_rewards = tf.placeholder(
                shape=[None, 1],  # Any number of elements, but of length 1
                dtype=tf.float32  # Only floats (rewards are clearly floats)
            )
            # The part of the net that will carry the past actions
            # Similarly to prev_rewards, it accepts any number of elements
            # but only integers (actions are encoded). Note that this time we
            # do not specify the [None, 1], cause... I don't know :-\ ...
            self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)

            # The timestep is simply a part of the net in charge of carrying
            # the 'termination flag' that in this case is simply the number of
            # timesteps.
            self.timestep = tf.placeholder(shape=[None, 1], dtype=tf.float32)

            # This will map the prev_actions to a one-hot representation of the
            # size of the total number of actions. The input is a tensor of
            # floats.
            self.prev_actions_onehot = tf.one_hot(
                self.prev_actions,
                a_size,
                dtype=tf.float32
            )

            # 1b. The hidden layer, i.e. the layer representing the learned
            #     policy. It is a concatenation of prev_rewards, prev_actions
            #     and timestep
            hidden = tf.concat(
                [self.prev_rewards, self.prev_actions_onehot, self.timestep],
                1
            )

            # 2. Recurrent network for temporal dependencies

            # We define our LSTM cell, as proposed by Wang and Duan. It will
            # store 48 units (any reason???).
            # state_is_tuple makes the net to map internal states as pair
            # (c_state, m_state)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(48, state_is_tuple=True)

            # The cell_state initialization: only zeros with floating point
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)

            # The hiddent state initialization: zeros with floating point
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)

            # We define a variable called state_init, wich is the pair (c, h)
            self.state_init = [c_init, h_init]

            # We define the inputs of the cell_state and hidden_state.
            # Both of them are float of shape [1, state_size]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])

            # We define the state_input as pair (c_in, h_in)
            self.state_in = (c_in, h_in)

            # The RNN input is basically the hidden state defined previously:
            #   prev_rewards, prev_actions, timestep
            # Inserts a dimension of 1 into a tensor's shape
            rnn_in = tf.expand_dims(hidden, [0])
            # step_size is 1 (I think...)
            step_size = tf.shape(self.prev_rewards)[:1]
            # state_in is basically the cell_state and hidden_state we defined
            # previously
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

            # We define the outputs and state:
            #   Using the lstm_cell, the rnn_in objects, the initialization of
            #   the state and the step_size (i.e. the sequence length) we can
            #   obtain the outputs layer and the state layer.
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell,
                rnn_in,
                initial_state=state_in,
                sequence_length=step_size,
                time_major=False
            )

            # We obtain the (c, h) of the obtained lstm_state
            lstm_c, lstm_h = lstm_state
            # We now define the state output_, which is basically the output
            # of the tuple
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            # The RNN output will be of shape (*, 48)
            rnn_out = tf.reshape(lstm_outputs, [-1, 48])

            # Actions, different that prev_actions. This will be the actual
            # output. Its declaration, however, is similar to the prev_actions
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(
                self.actions,
                a_size,
                dtype=tf.float32
            )

            # 3. Output layers for policy and value estimations

            # The policy is a fully-connected layer after the RNN output, with
            # a softmax of the actions size.
            self.policy = tf.contrib.layers.fully_connected(
                rnn_out,
                a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=helper.normalized_columns_initializer(.01),
                biases_initializer=None
            )
            # The value estimations (I guess similar to the Q-value) is a
            # fully-connected of size 1, with no activation.
            self.value = tf.contrib.layers.fully_connected(
                rnn_out,
                1,
                activation_fn=None,
                weights_initializer=helper.normalized_columns_initializer(1.0),
                biases_initializer=None
            )

            # Only the worker network need ops for loss functions and gradient
            # updating.
            if scope != 'global':
                # The target is the value: a stream of floats.
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                # There is something called advantages, which I do not know
                # what the fuck it is.
                self.advantages = tf.placeholder(
                    shape=[None],
                    dtype=tf.float32
                )
                # Responsible outputs is... ???????
                self.responsible_outputs = tf.reduce_sum(
                    self.policy * self.actions_onehot,
                    [1]
                )

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(
                    tf.square(self.target_v - tf. reshape(self.value, [-1]))
                )
                self.entropy = - tf.reduce_sum(
                    self.policy * tf.log(self.policy + 1e-7)
                )
                self.policy_loss = -tf.reduce_sum(
                    tf.log(self.responsible_outputs + 1e-7) * self.advantages
                )
                self.loss = 0.5 * self.value_loss + \
                    self.policy_loss - \
                    self.entropy * 0.05

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope
                )

                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(
                    self.gradients, 50.0
                )

                # Apply local gradients to global network
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    'global'
                )
                self.apply_grads = trainer.apply_gradients(
                    zip(grads, global_vars)
                )


class Worker():
    """The Worker is the controller."""

    def __init__(self, game, name, a_size, trainer, model_path,
                 global_episodes):
        """Construct the Worker."""
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        # Create the local copy of the network and the tensorflow op to copy
        # global paramters to local network
        self.local_AC = ACNet(a_size, self.name, trainer)
        self.update_local_ops = helper.update_target_graph('global', self.name)
        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        actions = rollout[:, 0]
        rewards = rollout[:, 1]
        timesteps = rollout[:, 2]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:, 4]

        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = helper.discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + \
            gamma * self.value_plus[1:] - \
            self.value_plus[:-1]
        advantages = helper.discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {
            self.local_AC.target_v: discounted_rewards,
            self.local_AC.prev_rewards: np.vstack(prev_rewards),
            self.local_AC.prev_actions: prev_actions,
            self.local_AC.actions: actions,
            self.local_AC.timestep: np.vstack(timesteps),
            self.local_AC.advantages: advantages,
            self.local_AC.state_in[0]: rnn_state[0],
            self.local_AC.state_in[1]: rnn_state[1]
        }

        v_l, p_l, e_l, g_n, v_n, _ = sess.run(
            [
                self.local_AC.value_loss,
                self.local_AC.policy_loss,
                self.local_AC.entropy,
                self.local_AC.grad_norms,
                self.local_AC.var_norms,
                self.local_AC.apply_grads
            ],
            feed_dict=feed_dict
        )

        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), \
            g_n, v_n

    def work(self, gamma, sess, coord, saver, train):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0

        print("Starting worker " + str(self.number))

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = [0, 0]
                episode_step_count = 0
                d = False
                r = 0
                a = 0
                t = 0
                self.env.reset()
                rnn_state = self.local_AC.state_init

                while d is False:
                    # Take an action using probabilities from policy network
                    # output.
                    a_dist, v, rnn_state_new = sess.run(
                        [
                            self.local_AC.policy,
                            self.local_AC.value,
                            self.local_AC.state_out
                        ], 
                        feed_dict={
                            self.local_AC.prev_rewards: [[r]],
                            self.local_AC.timestep: [[t]],
                            self.local_AC.prev_actions: [a],
                            self.local_AC.state_in[0]: rnn_state[0],
                            self.local_AC.state_in[1]: rnn_state[1]
                        }
                    )
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    rnn_state = rnn_state_new
                    r, d, t = self.env.pullArm(a)
                    episode_buffer.append([a, r, t, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_frames.append(
                        helper.set_image_bandit(
                            episode_reward,
                            self.env.bandit,
                            a,
                            t
                        )
                    )
                    episode_reward[a] += r
                    total_steps += 1
                    episode_step_count += 1

                self.episode_rewards.append(np.sum(episode_reward))
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of
                # the episode.
                if len(episode_buffer) != 0 and train is True:
                    v_l, p_l, e_l, g_n, v_n = self.train(
                        episode_buffer,
                        sess,
                        gamma,
                        0.0
                    )

                # Periodically save gifs of episodes, model parameters, and
                # summary statistics.
                if episode_count % 50 == 0 and episode_count != 0:
                    if episode_count % 500 == 0 and self.name == 'worker_0' \
                                                            and train is True:
                        saver.save(
                            sess,
                            self.model_path + '/model-' + str(episode_count) +
                            '.cptk'
                        )
                        print("Saved Model")

                    if episode_count % 100 == 0 and self.name == 'worker_0':
                        self.images = np.array(episode_frames)
                        helper.make_gif(
                            self.images,
                            './frames/image' + str(episode_count) + '.gif',
                            duration=len(self.images) * 0.1,
                            true_image=True,
                            # salience=False
                        )

                    # mean_reward = np.mean(self.episode_rewards[-50:])
                    # mean_length = np.mean(self.episode_lengths[-50:])
                    # mean_value = np.mean(self.episode_mean_values[-50:])
                    # summary = tf.Summary()
                    # summary.value.add(
                    #     tag='Perf/Reward',
                    #     simple_value=float(mean_reward)
                    # )
                    # summary.value.add(
                    #     tag='Perf/Length',
                    #     simple_value=float(mean_length)
                    # )
                    # summary.value.add(
                    #     tag='Perf/Value',
                    #     simple_value=float(mean_value)
                    # )
                    # if train is True:
                    #     summary.value.add(
                    #         tag='Losses/Value Loss',
                    #         simple_value=float(v_l)
                    #     )
                    #     summary.value.add(
                    #         tag='Losses/Policy Loss',
                    #         simple_value=float(p_l)
                    #     )
                    #     summary.value.add(
                    #         tag='Losses/Entropy',
                    #         simple_value=float(e_l)
                    #     )
                    #     summary.value.add(
                    #         tag='Losses/Grad Norm', simple_value=float(g_n)
                    #     )
                    #     summary.value.add(
                    #         tag='Losses/Var Norm', simple_value=float(v_n)
                    #     )
                    # self.summary_writer.add_summary(summary, episode_count)

                    # self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)

                episode_count += 1
