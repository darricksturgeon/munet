import tensorflow as tf


def parse(serialized):
    """
    reads tf record bytes into data
    :param serialized: byte string from tf record.
    :return: image and label tensors
    """

    features = \
        {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image']
    label_raw = parsed_example['label']

    # Decode the raw bytes to tensor images
    image = tf.decode_raw(image_raw, tf.float32)
    label = tf.decode_raw(label_raw, tf.int32)

    # The image and label are now correct TensorFlow types.
    return image, label


def batch_preprocess(image_batch, label_batch):
    """
    runs preprocessing (input distortions) on training data
    :param image_batch: tensor of inputs
    :param label_batch: tensor of labels
    :return:
    """
    shape = [-1, 90, 60, 60]
    image_batch = tf.reshape(image_batch, shape=shape)
    image_batch = tf.map_fn(lambda img: preproc(img), image_batch)
    image_batch, label_batch = tf.map_fn(
        lambda img, lab: distort(img, lab),
        tf.stack(image_batch, label_batch)
    )

    return image_batch, label_batch


def input_fn(record_path, train=True, batch_size=32, buffer_size=512):
    """
    batch input function
    :param record_path: path to tfrecord binary file created by pack.py
    :param train: boolean if is training data or false for testing data
    :param batch_size: num images to run per epoch
    :param buffer_size: shuffling size for training
    :return: image_batch and label_batch tf tensor iterators.
    """
    # tf record binary file
    dataset = tf.data.TFRecordDataset(filenames=record_path)

    # parse data on iteration
    dataset = dataset.map(parse)

    if train:
        # If training then read a buffer of the given size and
        # randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=buffer_size)

        # Allow infinite reading of the data.
        num_repeat = None
    else:
        # If testing then don't shuffle the data.
        num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels
    image_batch, label_batch = iterator.get_next()

    if train:
        image_batch = preproc(image_batch)
        image_batch, label_batch = distort(image_batch, label_batch)

    # The input-function must return a dict wrapping the images.
    x = {'x': image_batch}
    y = label_batch

    return x, y


def tf_input_training():
    """
    simple wrapper for input_fn because tf Estimators expect an input
    function which accepts no arguments.
    :return: input_fn() result (tf tensor iterator)
    """
    return input_fn('niftis.tfrecord', train=True, batch_size=32,
                    buffer_size=512)


def tf_input_testing():
    """
    simple wrapper for input_fn for testing data.
    :return: input_fn() result (tf tensor iterator)
    """
    return input_fn('niftis.tfrecord', train=False, batch_size=32,
                    buffer_size=512)


def _random_preprocessing(cfg):
    # returns preprocessing steps based on configuration.
    def _flip_lr(img, lab):
        img = tf.reverse(img, axis=[0])
        img = tf.reverse(img, axis=[0])
        lab = tf.reverse(lab, axis=[0])
        lab = tf.map_fn(
            _fs_lr_map.lookup, lab
        )
        return img, lab

    def _preproc(img):
        # reshape tensor, apply image operations and reshape back
        image_shape = tf.shape(img)
        img = tf.reshape(img, cfg['image_reshape'])
        img = tf.image.random_contrast(img, cfg['contrast'][0],
                                       cfg['contrast'][1])
        img = tf.image.random_brightness(img, cfg['brightness'][0],
                                         cfg['brightness'][1])
        # reshape
        img = tf.reshape(img, image_shape)
        # random gaussian noise @HARDCODED sigma
        img = img + tf.random_normal(image_shape, mean=cfg['noise'][0],
                                     stddev=cfg['noise'][1],
                                     dtype=tf.float32)
        # random rician noise
        # tf.sqrt(tf.random_normal() + tf.random_normal())

        return img

    def _distort(img, lab):
        def flip_lr(): return _flip_lr(img, lab)

        def no_flip(): return img, lab

        # random left-right flip and relabel at some rate...  Below is a
        # tensorflow if-then statement.
        img, lab = tf.case(
            [(tf.less(tf.random_uniform([], 0, 1), cfg['flip']), flip_lr)],
            default=no_flip)
        # @TODO random affine boost (mostly scaling)

        # @TODO random elastic distortion

        return img, lab

    return _preproc, _distort


_fs_lr_map = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(
        [2, 41, 3, 42, 4, 43, 5, 44, 6, 45, 7, 46, 8, 47, 9, 48, 10, 49, 11,
         50, 12, 51, 13, 52, 17, 53, 18, 54, 19, 55, 20, 56, 25, 57, 26, 58,
         27, 59, 28, 60, 29, 61, 30, 62, 31, 63, 32, 64, 33, 65, 34, 66, 35,
         67, 36, 68, 37, 69, 38, 70, 39, 71],
        [41, 2, 42, 3, 43, 4, 44, 5, 45, 6, 46, 7, 47, 8, 48, 9, 49, 10, 50,
         11, 51, 12, 52, 13, 53, 17, 54, 18, 55, 19, 56, 20, 57, 25, 58, 26,
         59, 27, 60, 28, 61, 29, 62, 30, 63, 31, 64, 32, 65, 33, 66, 34, 67,
         35, 68, 36, 69, 37, 70, 38, 71, 39]  # freesurfer left-right mapping
    ), -1
)

config = dict(contrast=(0.5, 2.0), brightness=(0.5, 2.0),  # rand ranges
              noise=(0.0, 4.0),
              flip=0.5,
              image_shape=[-1, 182, 218, 182],
              image_reshape=[-1, 182, 39676, 1],
              affine='twelve degrees of freedom or three (scaling)?')

preproc, distort = _random_preprocessing(config)
