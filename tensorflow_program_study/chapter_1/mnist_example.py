# encoding: utf-8
'''
@author: huaiyu-LI
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: lee1012116@126.com
@software: pycharm
@file: mnist_example.py
@time: 18-12-13 下午9:46
@desc:
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

path = os.path.split(os.path.realpath(__file__))[0]

# path = os.getcwd()
data_path = os.path.join(path, 'MNIST_data/')


def train():
    mnist = input_data.read_data_sets(data_path, one_hot=True)
    # 输入输出占符
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784])
    with tf.name_scope('output'):
        y_ = tf.placeholder(tf.float32, [None, 10])
    # 输入权重矩阵 和权重矩'
    with tf.name_scope('parameters'):
        with tf.name_scope('Weights'):
            W = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32, name="Weights")
            tf.summary.histogram('weights', W)
        with tf.name_scope('biase'):
            biase = tf.Variable(tf.zeros([10]), dtype=tf.float32, name="biase")
            tf.summary.histogram('biase', biase)
    with tf.name_scope('y_prediction'):
        y = tf.nn.softmax(tf.matmul(x, W) + biase)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
        tf.summary.histogram('loss', loss)
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.name_scope('init'):
        init = tf.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    merged_summary_op = tf.summary.merge_all()

    # summary_writer = tf
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("logs/", sess.graph)
        for _in in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})
            rs = sess.run(merged_summary_op, feed_dict={x: batch_xs, y_: batch_ys})
            writer.add_summary(rs, _in)
            if _in % 100 == 0:
                print("loss: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))

        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    train()
