# coding:utf-8
"""
mnist 数据集: 70,000 张黑底白字手写数字图片，其中 55,000 张为训练集， 5000 张为验证集，10,000 张为测试集。
    每张图片大小为 28*28=784 像素，图片中纯黑 色像素值为 0，纯白色像素值为 1。
    数据集的标签是长度为 10 的一维数组，数组中每个元素索引号表示对应数字出现的概率。

反向传播文件修改图片标签获取的接口, 关键操作: 利用多线程提高图片和标签的批获取效率

"""
import tensorflow as tf
import mnist_forward
import os
import mnist_generateds

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"
train_num_examples = 60000  # 之前用 mnist.train.num_examples 表示总样本数; 现在手动给出训练的总样本数 60,000


def backward():
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x, REGULARIZATION_RATE)  # 前向传播搭建的网络
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))  # 返回指定维度 axis 下，参数 x 中最大值索引号
    cem = tf.reduce_mean(ce)    # 取平均值
    loss = cem + tf.add_n(tf.get_collection('losses'))  # tf.get_collection(“”)函数表示从 collection 集合中取出全部变量生成一个列表。

    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 定义反向传播方法：含正则化
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在模型训练时引入滑动平均可以使模型在测试数据上表现的更加健壮
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver(max_to_keep=60)  # 实例化saver对象
    img_batch, label_batch = mnist_generateds.get_tfRecord(BATCH_SIZE, isTrain=True)

    with tf.Session() as sess:
        # 初始化所有模型参数
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 利用多线程提高图片和标签的批获取效率
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 训练模型
        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if step % 1000 == 0:
                print("After %d training step(s), loss on all data is %g " % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    backward()
