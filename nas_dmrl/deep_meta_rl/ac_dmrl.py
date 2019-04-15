"""The 'Learning to reinforcement learn' approach by Wang et al.

Here, we implment deep meta-reinforcement learning as described by Wang et al.
in their paper 'Learning to reinforcement learn', which is available at
https://arxiv.org/abs/1611.05763. We note that Wang's idea is similar to that
of 'RL$^2$: Fast Reinforcement Learning via Slow Reinforcement Learning', by
Duan et al.
"""

import threading
import numpy as np
import tensorflow as tf
# N_INPUTS = 1
FRAMEWORK_TF = 'tensorflow'


class PolicyRNN:
    """Recurrent Neural Network for RL^2.

    This class assumes that the input of the network is the 'embedded'
    representation of the (s, a, r, d) tuple that is received at each timestep.
    """

    def __init__(self, n_actions, state_shape, n_neurons=128,
                 framework='tensorflow'):
        """Initialize the network."""
        # Usual encapsulation in Python
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.n_neurons = n_neurons
        self.framework = framework

        # Build the network using the indicated framework
        if framework == FRAMEWORK_TF:
            self._build_tf_network()
        else:
            raise ValueError(
                "Framework '{val}' not supported yet.".format(val=framework)
            )

    def _build_tf_network(self):
        # Get the input layers
        in_current_state, in_past_reward, in_past_action, in_timestep = \
            self._tf_input_layers()
        # Convert the action to a one-hot encoding
        in_past_action_one_hot = tf.one_hot(
            indices=in_past_action,
            depth=self.n_actions,
            dtype=tf.float32,
            name="prev_action_onehot"
        )
        in_all_grouped = tf.concat(
            values=[
                in_current_state,
                in_past_reward,
                in_past_action_one_hot,
                in_timestep
            ],
            axis=1,
            name="all_inputs"
        )
        # Build the output
        rnn_input = tf.expand_dims(
            input=in_all_grouped,
            axis=[0],
            name="all_inputs_expanded"
        )

        # Build the recurrent unit
        lstm_cell = tf.keras.layers.LSTMCell(
            units=self.n_neurons,
            name="lstm_cell")

        recurrent_layer = tf.keras.layers.RNN(
            cell=lstm_cell,
            return_state=True,
            name="recurrent_layer"
        )

        rnn_result = recurrent_layer(rnn_input)
        rnn_output = rnn_result[0]
        # self.state_out = [rnn_output[1], rnn_output[2]]

        out_actions = tf.placeholder(
            shape=[None],
            dtype=tf.int32,
            name="next_action"
        )
        out_actions_onehot = tf.one_hot(
            indices=out_actions,
            depth=self.n_actions,
            dtype=tf.float32,
            name="next_action_onehot"
        )

        # Finally assign the network to a class attribute
        self.next_action_onehot = out_actions_onehot
        self.rnn_output = rnn_output

    def _tf_input_layers(self):
        # Create the input for the current_state
        new_shape = [None]
        new_shape.extend((x for x in self.state_shape))
        in_current_state = tf.placeholder(
            shape=new_shape,
            dtype=tf.float32,
            name="current_state"
        )
        # Create the input for the past reward
        in_past_reward = tf.placeholder(
            shape=[None, 1],
            dtype=tf.float32,
            name="past_reward"
        )
        # Create the input for the past action
        in_past_action = tf.placeholder(
            shape=[None],
            dtype=tf.int32,
            name="past_action"
        )
        # Create the input for the timestep (indicates the termination flag)
        in_timestep = tf.placeholder(
            shape=[None, 1],
            dtype=tf.float32,
            name="timestep"
        )
        return in_current_state, in_past_reward, in_past_action, in_timestep


class ActorCriticNetworkFactory:
    """Create Actor and Critic Networks, based on a given Policy."""

    def __init__(self):
        """Initialize the network."""

    def build_actor_network(policy=None):
        """Build the actor network.

        In the Actor-Critic setting, the Actor Network controls how the agent
        behaves. It is the actual learner of the policy and, during training,
        it receives a value function from the Critic.
        """
        if not isinstance(policy, PolicyRNN):
            raise TypeError("Policy must be of type PolicyRNN")
        if policy is None:
            raise ValueError("Policy cannot be None.")

        policy_layer = tf.contrib.layers.fully_connected(
            policy.rnn_output,
            policy.n_actions,
            activation_fn=tf.nn.softmax,
            biases_initializer=None
        )
        return policy_layer

    def build_critic_network(policy=None):
        """Build the critic network.

        In the Actor-Critic setting, the Critic Network measures how good the
        action taken by the Actor is. It approximates the value function and,
        during training, sends it to the Actor.
        """
        if not isinstance(policy, PolicyRNN):
            raise TypeError("Policy must be of type PolicyRNN")
        if policy is None:
            raise ValueError("Policy cannot be None.")

        value_layer = tf.contrib.layers.fully_connected(
            policy.rnn_output,
            1,
            activation_fn=None,
            biases_initializer=None
        )
        return value_layer


class ActorCriticMetaLearner:
    """Perform the training of the Actor-Critic Networks using deep Meta-RL."""

    def __init__(self, gamma, storage_path, max_time, rl_task,
                 ac_version="ac3"):
        """Initialize the object."""
        self.gamma = gamma
        self.storage_path = storage_path
        self.max_time = max_time
        self.episode_count = 0
        self.ac_version = ac_version
        self.rl_task = rl_task
        self.actors = None
        self.critic = None
        self.T = 0
        self.T_max = 100


    def _get_critic(self):
        with tf.variable_scope(name_or_scope="actor"):
            policy_rnn = PolicyRNN(
                n_actions=3,
                state_shape=[28],
                n_neurons=128,
            )
            actor = ActorCriticNetworkFactory.build_critic_network(
                policy=policy_rnn
            )

            return actor

    def _get_actors(self, n_critics=5):
        critics = []
        for i in range(n_critics):
            with tf.variable_scope(name_or_scope="actor_{idx}".format(idx=i)):
                policy_rnn = PolicyRNN(
                    n_actions=3,
                    state_shape=[28],
                    n_neurons=128,
                )
                critics.append(
                    ActorCriticNetworkFactory.build_actor_network(
                        policy=policy_rnn
                    )
                )

        return critics

    def training_step(self):
        """Perform the training with AC deep meta-reinforcement learning."""

    def _actor_update(self):
        """"""

    def _critic_update(self):
        """"""

    def actor_learner(self, actor):
        # Assume global shared parameter vectors θ and θ_v
        # and global shared counter T = 0
        # Assume thread-specific parameter vectors θ' and θ'_v

        t = 1  # Initialize thread step counter t
        t_max = 100
        while self.T > self.T_max:
            # Reset gradients dθ ← 0 and dθv ← 0.
            # Synchronize thread-specific parameters θ' = θ and θ'v = θv
            t_start = t

            # Get state st
            while is_terminal(state_t) or (t - t_start) == t_max:
                reward, next_state = execute(a_t, policy)
                t += 1
                self.T += self.T

            # Obtain R following equation

            for i in range(t-1, t_start + 1): # Check boundaries
                R = r_i + self.gamma*R
                # Accumulate gradients
            # Perform asynchronous update
            self.update_global_network(new_params)

    def learn(self, n_critics=5):
        """Perform the actual training."""
        # For the train we need to have n_critics, one actor and perform the
        # iteration
        self.actor = self._get_actor()
        self.critics = self._get_critics(n_critics=n_critics)

        for trial in range(self.task.n_trials):
            # Reset the hiddent state of the networks
            # Resample the MDP
            for episode in range(self.task.n_episodes):
                # Run the critics

                # Train the actor

                # Update the critics
