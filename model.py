import tensorflow as tf


def model_fn(features, labels, mode, params):
    """
    model function for custom tf Estimator
    :param features: input training data
    :param labels: input label data
    :param mode: tf mode
    :param params: special input parameters for creating the model
    :return: tf model graph
    """

    train = (mode == tf.estimator.ModeKeys.TRAIN)

    # network architecture
    x = tf.reshape(features['x'], params['shape'])

    lvl1 = tf.layers.conv3d(x, filters=16, kernel_size=3, padding='valid',
                            activation=tf.nn.relu)
    lvl1 = tf.layers.dropout(lvl1, rate=0.2, training=train)
    net = tf.layers.max_pooling3d(lvl1, pool_size=2, strides=2, padding='valid')

    lvl2 = tf.layers.conv3d(net, filters=32, kernel_size=3, padding='valid',
                            activation=tf.nn.relu)
    net = tf.layers.max_pooling3d(lvl2, pool_size=2, strides=2, padding='valid')

    lvl3 = tf.layers.conv3d(net, filters=64, kernel_size=3, padding='valid',
                            activation=tf.nn.relu)
    net = tf.layers.max_pooling3d(lvl3, pool_size=2, strides=2, padding='valid')

    net = tf.layers.conv3d(net, filters=128, kernel_size=3, padding='valid',
                           activation=tf.nn.relu)
    net = tf.layers.conv3d(net, filters=128, kernel_size=3, padding='valid',
                           activation=tf.nn.relu)

    net = tf.layers.conv3d_transpose(net, kernel_size=2, strides=2,
                                     padding='valid')
    net = tf.layers.conv3d(net, filters=64, kernel_size=3, padding='valid',
                           activation=tf.nn.relu)

    net = tf.layers.conv3d_transpose(net, kernel_size=2, strides=2,
                                     padding='valid')
    net = tf.layers.conv3d(net, filters=32, kernel_size=3, padding='valid',
                           activation=tf.nn.relu)

    net = tf.layers.conv3d_transpose(net, kernel_size=2, strides=2,
                                     padding='valid')
    net = tf.layers.conv3d(net, filters=16, kernel_size=3, padding='valid',
                           activation=tf.nn.relu)

    net = tf.layers.conv3d(net, filters=4, kernel_size=1, padding='full')

    if mode == tf.estimator.ModeKeys.PREDICT:
        pass
    else:
        pass

    return net
