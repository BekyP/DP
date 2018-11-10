import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


def print_shape(tensor):
    print(tensor.get_shape)


# creates model of NN, data = input image, keep_prob = value on dropout layer
def neural_network_model(data, keep_prob):
    with tf.name_scope('convolution_layer_1'):
        w_conv = tf.Variable(tf.truncated_normal(
            [5, 5, 3, 32], stddev=0.1)
        )
        b_conv = tf.Variable(tf.zeros(shape=[32]))

        x_image = tf.reshape(data, [-1, 64, 64, 3])
        conv = tf.nn.sigmoid(
            tf.nn.conv2d(x_image, w_conv,
                         strides=[1, 1, 1, 1], padding='SAME') + b_conv
                             )
    print_shape(conv)

    with tf.name_scope('max_pooling_layer_1'):
        pool = tf.nn.max_pool(
            conv, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME'
        )
    print_shape(pool)

    with tf.name_scope('normalization_layer'):
        norm = tf.reshape(pool, [-1, 32 * 32 * 32])
    print_shape(norm)

    with tf.name_scope('fully_connected_layer_1'):
        l1 = fully_connected(norm, 64 * 64, activation_fn=tf.nn.sigmoid)
    print_shape(l1)

    with tf.name_scope('dropout_layer'):
        drop = tf.nn.dropout(l1, keep_prob)

    with tf.name_scope('output_layer'):
        output = fully_connected(drop, 64 * 64, activation_fn=None)

    with tf.name_scope('reshaped_output_layer'):
        output = tf.reshape(output, [-1, 64, 64])
    print_shape(output)

    return output
