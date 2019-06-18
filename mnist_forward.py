# coding:utf-8
import tensorflow as tf
__author__ = 'Yixu Wang'

INPUT_NODE = 784
LAYER1_NODE = 500
OUTPUT_NODE = 10


def get_weight(shape, regularizer):
    """对参数 w 的设置，包括参数 w 的形状和是否正则化的标志
    """
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    """对参数 b 的设置
    """
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    """前向传播：定义神经网络中的参数 w 和偏置 b，定义由输入到输出的网络结构
    """
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2  # 输出层不激活
    return y
