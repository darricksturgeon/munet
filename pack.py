import argparse
import os

import nibabel

import numpy as np
import tensorflow as tf


def cli_interface():
    parser = argparse.ArgumentParser('load niftis and labels into tfrecord '
                                     'format')

    parser.add_argument('--record', '-r',
                        help='name of the output record file.')
    parser.add_argument('--inputs', '-i',
                        help='list of paths or text file containing paths to '
                             'input images.')
    parser.add_argument('--labels', '-l', help='label data')
    parser.add_argument('--split', '-s',
                        help='ratio of train/test data or predetermined split '
                             '(to be implemented)')
    parser.add_argument('--shuffle', type=bool,
                        help='shuffle inputs before writing to tfrecord.')

    args = parser.parse_args()

    return interface(**vars(args))


def interface(record, inputs, labels, split=.5, shuffle=False, save_dims=False,
              **kwargs):

    with open(inputs) as fd:
        inputs = fd.readlines()
    with open(labels) as fd:
        labels = fd.readlines()

    # get data properties
    if save_dims:
        #@TODO how to pass this information to tensorflow later?
        # perhaps a config file for the run...
        nii = nibabel.load(inputs[0].strip())
        dims = nii.shape
        header = nii.header

    create_tfrecord_imgsegs(record, inputs, labels)


def create_tfrecord_imgsegs(name, inputs, labels):
    # creates record file assuming labels are also images.
    if os.path.exists(name):
        os.remove(name)

    with tf.python_io.TFRecordWriter(name) as writer:
        for filename, lab in zip(inputs, labels):
            nii = nibabel.load(filename)
            lab = nibabel.load(lab)

            # load images as bytes
            # @WARNING enforcing type conventions here, since data must be
            # binarized
            img = nii.get_data().astype(np.float32).tostring()
            lab = lab.get_data().astype(np.int32).tostring()

            # wrap data into tf Example object
            feature = tf.train.Features(feature={
                'image': wrap_bytes(img),
                'label': wrap_bytes(lab),
            })
            example = tf.train.Example(features=feature)

            # Save current input into tfrecords file
            writer.write(example.SerializeToString())


def distort(img):
    img = tf.image.random_hue(img, max_delta=0.05)
    img = tf.image.random_contrast(img, lower=0.3, upper=1.0)
    img = tf.image.random_saturation(img, lower=0.0, upper=2.0)
    img = tf.image.random_flip_left_right(img)

    return img


def distort_batch(image_batch: tf.Tensor) -> tf.Tensor:
    shape = [-1, 500, 500, 3]  # @hardcoded Cifar10
    image_batch = tf.reshape(image_batch, shape=shape)
    image_batch = tf.map_fn(lambda img: distort(img), image_batch)

    return image_batch


def wrap_bytes(value):
    # converts raw image data into tf object
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    cli_interface()
