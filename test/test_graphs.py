"""Module to test the model builder."""

import unittest
import nas_dmrl.deep_meta_rl.ac_dmrl as dmrl
import tensorflow as tf

WORKSPACE_DIR = "./workspace"
GRAPHS_DIR = "{workspace}/graph".format(workspace=WORKSPACE_DIR)

class TestGraphs(unittest.TestCase):
    """Test the Assistan features."""

    def test_policy_graph(self):
        tf.reset_default_graph()

        with tf.variable_scope(name_or_scope="policy"):
            dmrl.PolicyRNN(n_actions=3, state_shape=[28], n_neurons=128)
            file_writer = tf.summary.FileWriter(GRAPHS_DIR,
                                                tf.get_default_graph())
            file_writer.close()

    def test_actor_graph(self):
        tf.reset_default_graph()

        with tf.variable_scope(name_or_scope="actor"):
            policy_rnn = dmrl.PolicyRNN(
                n_actions=3,
                state_shape=[28],
                n_neurons=128,
            )
            dmrl.ActorCriticNetworkFactory.build_actor_network(policy_rnn)
            file_writer = tf.summary.FileWriter(GRAPHS_DIR,
                                                tf.get_default_graph())
            file_writer.close()

    def test_critic_graph(self):
        tf.reset_default_graph()

        with tf.variable_scope(name_or_scope="critic"):
            policy_rnn = dmrl.PolicyRNN(
                n_actions=3,
                state_shape=[28],
                n_neurons=128,
            )
            dmrl.ActorCriticNetworkFactory.build_critic_network(policy_rnn)
            file_writer = tf.summary.FileWriter(GRAPHS_DIR,
                                                tf.get_default_graph())
            file_writer.close()
