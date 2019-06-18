# coding:utf-8
import tensorflow as tf
from PIL import Image
import os

image_train_path = './mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path = './mnist_data_jpg/mnist_train_jpg_60000.txt'
image_test_path = './mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path = './mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_train = './data/mnist_train.tfrecords'
tfRecord_test  = './data/mnist_test.tfrecords'
data_path='./data'


def write_tfRecord(tfRecordName, image_path, label_path):
    '''生成tfrecords文件。tfrecords 是一种二进制文件，可先将图片和标签制作成该格式的文件。
    使用 tfrecords 进行数据读取，会提高内存利用率。
    '''
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()
    # 循环遍历每张图和标签
    for content in contents:
        value = content.split()
        img_path = image_path + value[0]
        img = Image.open(img_path)
        img_raw = img.tobytes()
        labels = [0] * 10
        labels[int(value[1])] = 1
        # 存入 example
        print(img)
        example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
            }))
        writer.write(example.SerializeToString())   # 将 example 序列化成字符串存储
        num_pic += 1
        print("the number of pics: ", num_pic)
    writer.close()
    print('write tfrecord successful')


def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.mkdir(data_path)
        print('The directory was created successfully')
    else:
        print('The directory already exists')
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)


# 解析 tfrecords 文件
def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)  # 生成一个队列，用来读取数据
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([10], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label


def get_tfRecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    # 随机读取一个 batch 的数据
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=num,
                                                    num_threads=2,
                                                    capacity=1000,
                                                    min_after_dequeue=700)
    return img_batch, label_batch


if __name__ == '__main__':
    generate_tfRecord()