"""The training module."""

import argparse
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from nas_rl2.nas.net_manager import NetManager
from nas_rl2.nas.controller import Reinforce


def parse_args():
    """Parse srguments of the scrip"""
    desc = "TensorFlow implementation of 'Neural Architecture Search with \
        Reinforcement Learning'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--max_layers', default=2)

    args = parser.parse_args()
    args.max_layers = int(args.max_layers)
    return args


def policy_network(state, max_layers):
    """Policy network is a main network for searching optimal architecture.

    It uses NAS - Neural Architecture Search recurrent network cell.
    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/rnn/
    python/ops/rnn_cell.py#L1363

    Args:
        state: current state of required topology
        max_layers: maximum number of layers
    Returns:
        3-D tensor with new state (new topology)

    """
    with tf.name_scope("policy_network"):
        nas_cell = tf.contrib.rnn.NASCell(4*max_layers)
        outputs, state = tf.nn.dynamic_rnn(
            nas_cell,
            tf.expand_dims(state, -1),
            dtype=tf.float32
        )
        bias = tf.Variable([0.05]*4*max_layers)
        outputs = tf.nn.bias_add(outputs, bias)
        print(
            "outputs: ",
            outputs,
            outputs[:, -1:, :],
            tf.slice(outputs, [0, 4*max_layers-1, 0], [1, 1, 4*max_layers])
        )
        # return tf.slice(outputs, [0, 4*max_layers-1, 0],[1, 1, 4*max_layers])
        # Returned last output of rnn
        return outputs[:, -1:, :]


def train(mnist):
    """Perform training on mnist dataset."""
    global args
    sess = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(
        0.99,
        global_step,
        500,
        0.96,
        staircase=True
    )

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    reinforce = Reinforce(
        sess,
        optimizer,
        policy_network,
        args.max_layers,
        global_step
    )
    net_manager = NetManager(num_input=784,
                             num_classes=10,
                             learning_rate=0.001,
                             mnist=mnist,
                             bathc_size=100)

    max_episodes = 2500
    step = 0
    state = np.array(
        [[10.0, 128.0, 1.0, 1.0]*args.max_layers],
        dtype=np.float32
    )
    pre_acc = 0.0
    total_rewards = 0
    for i_episode in range(max_episodes):
        action = reinforce.get_action(state)
        print("ca:", action)
        if all(ai > 0 for ai in action[0][0]):
            reward, pre_acc = net_manager.get_reward(action, step, pre_acc)
            print("=====>", reward, pre_acc)
        else:
            reward = -1.0
        total_rewards += reward

        # In our sample action is equal state
        state = action[0]
        reinforce.storeRollout(state, reward)

        step += 1
        ls = reinforce.train_step(1)
        log_str = "current time:  " + str(datetime.datetime.now().time()) + \
            " episode:  " + str(i_episode) + " loss:  " + str(ls) + \
            " last_state:  " + str(state) + " last_reward:  " + str(reward) + \
            "\n"
        log = open("lg3.txt", "a+")
        log.write(log_str)
        log.close()
        print(log_str)


def main():
    """Execute main process"""
    global args
    args = parse_args()

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
