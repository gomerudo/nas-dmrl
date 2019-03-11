"""The Controller in NAS.

Input:
- The current state (in this task, state and action are the same things).
- Maximum number of searching layers.

Output:
- New action to update the desired neural network.

Every `action` depends on the previous state, but sometimes, for more effective
training, we can generate random actions to avoid local minimums.

In each cycle, our network will generate an `action`, get `rewards`, and after
that, take a training step.

"""

import random
import numpy as np
import tensorflow as tf


class Reinforce():
    """This will train the controller or something like that..."""

    def __init__(self, sess, optimizer, policy_network, max_layers,
                 global_step, division_rate=100.0, reg_param=0.001,
                 discount_factor=0.99, exploration=0.3):
        """Constructor.

        It takes three types of parameters:
        - TensorFlow objects: sess, optimizer
        - Reinforcement Learning parameters: policy_network, discount_factor,
          exploration
        - Search constraints: max_layers, division_rate, reg_param

        This __init__ step consists of simple variable encapsulation and
        initialization of global tensor flow variables.

        Params:
            sess (tf.session) An already initialized TensorFlow session.
            optimizer (tf.optimizer) An already initialized TensorFlow session,
                which is separetely initialized from `sess`.
            policy_network (?) The method initialized above (???)
            max_layers (int) The maximum number of layers allowed for the
                resulting network.
            global_step (?)
            division_rate (float) Values, from a normal distribution, of each
                neuron. Ranges in [-1.0, 1.0].
            reg_param (float) The regularization parameter.
            discount_factor (float) The discount factor used in Reinforcement
                Learning.
            exploration (float) The probability of generating a random action,
                i.e. the epsilon.
        """
        # Simple encapsulation of the parameters
        self.sess = sess
        self.optimizer = optimizer
        self.policy_network = policy_network
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor = discount_factor
        self.max_layers = max_layers
        self.global_step = global_step
        self.exploration = exploration

        # Buffers for rewards and states (?????)
        self.reward_buffer = []
        self.state_buffer = []

        # We call the create_variables() method to build the layers used for
        # input and outputs (prediction)
        self.create_variables()

        # From documentation:
        #
        # Global variables are variables that are shared across machines in a
        # distributed environment (model variables are subset of these).
        # Commonly, all TRAINABLE_VARIABLES variables will be in
        # MODEL_VARIABLES, and all MODEL_VARIABLES variables will be in
        # GLOBAL_VARIABLES.
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # The method `variables_initializer` returns an Op that initializes a
        # list of variables. 
        # run simply executes the initialization defined in the Op above.
        self.sess.run(tf.variables_initializer(var_lists))

    def get_action(self, state):
        """Get the action to perform based on a current state."""
        return self.sess.run(self.predicted_action, {self.states: state})

        # This is, for some reason, skipped. Maybe the tutorial explains why.
        if random.random() < self.exploration:
            return np.array([[random.sample(range(1, 35), 4*self.max_layers)]])
        else:
            return self.sess.run(self.predicted_action, {self.states: state})

    def create_variables(self):
        """Create placeholders for model's layers."""
        # Remember that tensorflow provides a 'scope' that will helps us to
        # create graphs: eg. /model_inputs/states.
        
        # 1. /model_inputs/**
        with tf.name_scope("model_inputs"):
            # raw state representation
            #   - Accept floats as states (why only floats???)
            #   - We accept stream of elements of max_layers*4.
            self.states = tf.placeholder(
                tf.float32,
                [None, self.max_layers*4],
                name="states"
            )

        # 2. /predict_actions/**
        #   What I think we do here is to use the policy network's output and
        #   then compute the predicted action based on the highest "q-value".
        with tf.name_scope("predict_actions"):
            # initialize policy network

            # /predict_actions/policy_network/**
            with tf.variable_scope("policy_network"):
                # Dunno yet...
                # /predict_actions/policy_network/policy_outputs
                self.policy_outputs = self.policy_network(
                    self.states,
                    self.max_layers
                )

            # /predict_actions/action_scores
            #   Uses the /predict_actions/policy_network/policy_outputs net
            #   to return an identity of the output - but as a tensor I guess.
            self.action_scores = tf.identity(
                self.policy_outputs,
                name="action_scores"
            )

            # /predict_actions/predicted_action
            #   Cast a scalar multiplication into an integer. I assume
            #   something like this is happening:
            #       int( max(division_rate*action_scores) )
            self.predicted_action = tf.cast(
                tf.scalar_mul(
                    self.division_rate,
                    self.action_scores
                ),
                tf.int32,
                name="predicted_action"
            )

        # 3. Regularization loss
        #   We will obtain the trainable variables only - they belong to the
        #   policy network scope.
        policy_network_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="policy_network"
        )

        # 4. compute loss and gradients: gradients for selecting action from
        # policy network
        #   /compute_gradients/**
        with tf.name_scope("compute_gradients"):
            # This is a layer for the discounted rewards. It will receive a
            # stream of float values (the rewards).
            self.discounted_rewards = tf.placeholder(
                tf.float32,
                (None,),
                name="discounted_rewards"
            )

            # /compute_gradients/policy_network/**
            # I am not sure about the consequence of the 'reuse' parameter
            with tf.variable_scope("policy_network", reuse=True):
                # We will use the policy_network again, and will obtain the
                # log probabilities.
                self.logprobs = self.policy_network(
                    self.states,
                    self.max_layers
                )
                print("self.logprobs", self.logprobs)

            # Here we compute the cross_entropy loss for the logprobs we
            # obtained, against the states (?)
            self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logprobs[:, -1, :],
                labels=self.states
            )

            # Now, we will obtain a tensor with the mean of values along both
            # axes.
            self.pg_loss = tf.reduce_mean(
                self.cross_entropy_loss
            )

            # Now we compute the regularization loss for each of the trainable
            # variables.
            self.reg_loss = tf.reduce_sum(
                [
                    tf.reduce_sum(tf.square(x))
                    for x in policy_network_variables
                ]
            )  # Regularization
            self.loss = self.pg_loss + self.reg_param * self.reg_loss

            # compute gradients
            self.gradients = self.optimizer.compute_gradients(self.loss)

            # compute policy gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)

            # training update
            with tf.name_scope("train_policy_network"):
                # apply gradients to update policy network
                self.train_op = self.optimizer.apply_gradients(
                    self.gradients,
                    global_step=self.global_step
                )

    def storeRollout(self, state, reward):
        """Store the state-reward pair in the buffer (list)."""
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])

    def train_step(self, steps_count):
        """Perform a single train step."""
        # States are the ones in the buffer
        states = np.array(self.state_buffer[-steps_count:])/self.division_rate
        # rewards are the ones in the buffer too
        rewards = self.reward_buffer[-steps_count:]
        # We simply perform a training step using the elements declared in
        # create_variables().
        _, res_loss = self.sess.run(
            [self.train_op, self.loss],
            {
                self.states: states,
                self.discounted_rewards: rewards
            }
        )
        return res_loss
