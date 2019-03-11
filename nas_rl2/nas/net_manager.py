"""Manage the reward obtained from the network."""

import tensorflow as tf
import tf.contrib.layers as tfl


class NetManager():
    """The class in charge of obtained the rewards for the network."""

    def __init__(self, num_input, num_classes, learning_rate, mnist,
                 max_step_per_action=5500*3,
                 bathc_size=100,
                 dropout_rate=0.85):
        """Constructor."""
        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.mnist = mnist

        self.max_step_per_action = max_step_per_action
        self.bathc_size = bathc_size
        self.dropout_rate = dropout_rate

    def get_reward(self, action, step, pre_acc):
        """Compute of the reward."""
        action = [action[0][0][x:x+4] for x in range(0, len(action[0][0]), 4)]
        cnn_drop_rate = [c[3] for c in action]

        with tf.Graph().as_default() as g:
            with g.container('experiment'+str(step)):
                model = CNN(self.num_input, self.num_classes, action)

                loss_op = tf.reduce_mean(model.loss)
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate
                )
                train_op = optimizer.minimize(loss_op)

                with tf.Session() as train_sess:
                    init = tf.global_variables_initializer()
                    train_sess.run(init)

                    for step in range(self.max_step_per_action):
                        batch_x, batch_y = self.mnist.train.next_batch(
                            self.bathc_size
                        )
                        feed = {model.X: batch_x,
                                model.Y: batch_y,
                                model.dropout_keep_prob: self.dropout_rate,
                                model.cnn_dropout_rates: cnn_drop_rate}
                        _ = train_sess.run(train_op, feed_dict=feed)

                        if step % 100 == 0:
                            # Calculate batch loss and accuracy
                            loss, acc = train_sess.run(
                                [loss_op, model.accuracy],
                                feed_dict={
                                    model.X: batch_x,
                                    model.Y: batch_y,
                                    model.dropout_keep_prob: 1.0,
                                    model.cnn_dropout_rates:
                                        [1.0]*len(cnn_drop_rate)
                                }
                            )
                            print(
                                "Step " + str(step) +
                                ", Minibatch Loss= " + "{:.4f}".format(loss) +
                                ", Current accuracy= " + "{:.3f}".format(acc)
                            )
                    batch_x, batch_y = self.mnist.test.next_batch(10000)
                    loss, acc = train_sess.run(
                        [loss_op, model.accuracy],
                        feed_dict={
                            model.X: batch_x,
                            model.Y: batch_y,
                            model.dropout_keep_prob: 1.0,
                            model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)
                        }
                    )
                    print("!!!!!!acc:", acc, pre_acc)

                    if acc - pre_acc <= 0.01:
                        return acc, acc
                    # implicit else
                    return 0.01, acc


class CNN():
    """The CNN used in the experiments."""

    def __init__(self, num_input, num_classes, cnn_config):
        """Constructor."""
        cnn = [c[0] for c in cnn_config]
        cnn_num_filters = [c[1] for c in cnn_config]
        max_pool_ksize = [c[2] for c in cnn_config]

        self.X = tf.placeholder(
            tf.float32,
            [None, num_input], 
            name="input_X"
        )
        self.Y = tf.placeholder(
            tf.int32,
            [None, num_classes],
            name="input_Y"
        )
        self.dropout_keep_prob = tf.placeholder(
            tf.float32,
            [],
            name="dense_dropout_keep_prob"
        )
        self.cnn_dropout_rates = tf.placeholder(
            tf.float32,
            [len(cnn), ],
            name="cnn_dropout_keep_prob"
        )

        Y = self.Y
        X = tf.expand_dims(self.X, -1)
        pool_out = X
        with tf.name_scope("Conv_part"):
            for idd, filter_size in enumerate(cnn):
                with tf.name_scope("L"+str(idd)):
                    conv_out = tf.layers.conv1d(
                        pool_out,
                        filters=cnn_num_filters[idd],
                        kernel_size=(int(filter_size)),
                        strides=1,
                        padding="SAME",
                        name="conv_out_"+str(idd),
                        activation=tf.nn.relu,
                        kernel_initializer=tfl.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer
                    )
                    pool_out = tf.layers.max_pooling1d(
                        conv_out,
                        pool_size=(int(max_pool_ksize[idd])),
                        strides=1,
                        padding='SAME',
                        name="max_pool_"+str(idd)
                    )
                    pool_out = tf.nn.dropout(
                        pool_out,
                        self.cnn_dropout_rates[idd]
                    )

            flatten_pred_out = tf.contrib.layers.flatten(pool_out)
            self.logits = tf.layers.dense(flatten_pred_out, num_classes)

        self.prediction = tf.nn.softmax(self.logits, name="prediction")
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=Y,
            name="loss"
        )
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(Y, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_pred, tf.float32),
            name="accuracy"
        )
