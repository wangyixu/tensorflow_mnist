# coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_forward
import mnist_backward


def pre_pic(picName):
    """预处理图片，包括resize、转变灰度图、二值化
    """
    img = Image.open(picName)
    reIm = img.resize((28,28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50  # 设定合理的阈值
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]   # 输入是白底黑字，模型要求黑底白字
            if im_arr[i][j] < threshold:    # 对图片做二值化处理（滤掉噪声）
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)  # 0-1之间的浮点数
    return img_ready


def restore_model(testPicArr):
    """复现视图 session ，对输入图片数组做出预测
    """
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)  # 得到概率最大的预测值

        # 实现滑动平均模型
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_to_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_to_restore)

        with tf.Session() as sess:
            # 加载训练好的模型
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            # 如果已有 ckpt 模型则恢复会话、轮数、计算准确率
            if ckpt is not None:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    preValue = sess.run(preValue, feed_dict={x: testPicArr})
                    return preValue
            else:
                print('No checkpoint file found')
                return -1


def application():
    testNum = input("input the number of test pictures:")
    for i in range(int(testNum)):
        testPic = input("the path of test picture:")
        testPicArr = pre_pic(testPic)   # 预处理
        preValue = restore_model(testPicArr)    # 用保存的模型预测
        print("The prediction number is:", preValue)


if __name__ == '__main__':
    application()