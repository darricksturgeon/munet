import tensorflow as tf


def parse(serialized):
    # these are the expected features in the dataset. I may need to modify if we end up trying segmentation.
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
    shape = [-1, 90, 60, 60]
    image_batch = tf.reshape(image_batch, shape=shape)
    image_batch = tf.map_fn(lambda img: preproc(img), image_batch)
    image_batch, label_batch = tf.map_fn(
        lambda img, lab: distort(img, lab),
        tf.stack(image_batch, label_batch)
    )

    return image_batch, label_batch


def input_fn(record_path, train=True, batch_size=32, buffer_size=512):
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
        image_batch, label_batch = preprocess(image_batch, label_batch)

    # The input-function must return a dict wrapping the images.
    x = {'x': image_batch}
    y = label_batch

    return x, y


def tf_input():
    pass


def _random_preprocessing(cfg):
    # returns preprocessing steps based on configuration.

    def preproc(img):
        # random contrast
        tf.random_uniform()
        # random saturation
        tf.random_uniform()
        # random gaussian noise
        tf.random_normal()
        # random rician noise
        # tf.sqrt(tf.random_normal() + tf.random_normal())

        return img

    def distort(img, lab):
        # random left-right flip and relabel at some rate...
        img = tf.reverse(img, axis=0)
        lab = tf.reverse(lab, axis=0)
        lab = tf.map_fn(
            lambda x: _fs_lr_map[x],
        )
        # random affine boost (mostly scaling)

        # random distortion

        return img, lab

    return preproc, distort

_fs_lr_map = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer()
)

config = {
    'random_contrast': (0.0, 2.0),
    'random_saturation': (0.0, 2.0),
    'random_noise': ('some parameter'),
    'random_flip': 0.5,
    'affine_parameters': ('twelve degrees of freedom?')
}

preproc, distort = _random_preprocessing(config)
