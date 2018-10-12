import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names


class VGG16():

    def __init__(self, image, path_weights):
        self.image = image
        self.data = np.load(path_weights)
        self.build()
        self.result = self.fc8

    def Convlayers(self, layers, name):
        with tf.name_scope(name) as scope:
            kernel = tf.constant(self.get_weights(name),
                                 name="weights")
            conv = tf.nn.conv2d(layers, kernel, [1, 1, 1, 1], padding="SAME")
            biases = tf.constant(self.get_biases(name),
                                 name="biases")
            out = tf.nn.bias_add(conv, biases)
            return tf.nn.relu(out, name=scope)

    def fc_layers(self, layers, name):
        with tf.name_scope(name) as scope:
            fcW = tf.constant(self.get_weights(name),
                              name="weights")
            fcb = tf.constant(self.get_biases(name),
                              name="biases")
            fcl = tf.nn.bias_add(tf.matmul(layers, fcW), fcb)
            if name == "fc8":
                return tf.nn.softmax(fcl)
            return tf.nn.relu(fcl)

    def max_pooling(self, layers, name):
        return tf.nn.max_pool(layers, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding="SAME", name=name)

    def get_weights(self, name):
        return self.data["{}_W".format(name)]

    def get_biases(self, name):
        return self.data["{}_b".format(name)]

    def build(self):
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        image = self.image - mean
        self.conv1_1 = self.Convlayers(image, "conv1_1")
        self.conv1_2 = self.Convlayers(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pooling(self.conv1_2, "pool1")

        self.conv2_1 = self.Convlayers(self.pool1, "conv2_1")
        self.conv2_2 = self.Convlayers(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pooling(self.conv2_2, "pool2")

        self.conv3_1 = self.Convlayers(self.pool2, "conv3_1")
        self.conv3_2 = self.Convlayers(self.conv3_1, "conv3_2")
        self.conv3_3 = self.Convlayers(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pooling(self.conv3_3, "pool3")

        self.conv4_1 = self.Convlayers(self.pool3, "conv4_1")
        self.conv4_2 = self.Convlayers(self.conv4_1, "conv4_2")
        self.conv4_3 = self.Convlayers(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pooling(self.conv4_3, "pool4")

        self.conv5_1 = self.Convlayers(self.pool4, "conv5_1")
        self.conv5_2 = self.Convlayers(self.conv5_1, "conv5_2")
        self.conv5_3 = self.Convlayers(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pooling(self.conv5_3, "pool5")

        shape = int(np.prod(self.pool5.get_shape()[1:]))
        pool5_flat = tf.reshape(self.pool5, [-1, shape])
        self.fc6 = self.fc_layers(pool5_flat, "fc6")
        self.fc7 = self.fc_layers(self.fc6, "fc7")
        self.fc8 = self.fc_layers(self.fc7, "fc8")


if __name__ == "__main__":
    session = tf.Session()
    image = imread("laska.png", mode="RGB")
    image = imresize(image, (224, 224))
    vgg = VGG16(image, "vgg16_weights.npz")
    prob = session.run(vgg.result)[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])
