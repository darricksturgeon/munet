import argparse
import os

import nibabel

import numpy as np
import tensorflow as tf


def cli_interface():
    """
    command line interface for pack.py
    :return: interface call
    """
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
    """
    creates a tensorflow record file given lists of niftis for images and
    labels (segmentations).
    :param record: name of record file to save to.
    :param inputs: newline delimited text file of image nifti paths.
    :param labels: newline delimited text file of label nifti paths.
    :param split: test/train split (NotYetImplemented)
    :param shuffle: shuffle inputs or not (NotYetImplemented)
    :param save_dims: save image dimensions into config file (NotYetImplemented)
    :param kwargs:
    :return: None
    """

    with open(inputs) as fd:
        inputs = fd.readlines()
    with open(labels) as fd:
        labels = fd.readlines()

    # get data properties
    if save_dims:
        # @TODO save info into a configuration for runtime?
        nii = nibabel.load(inputs[0].strip())
        dims = nii.shape
        header = nii.header

    create_tfrecord_imgsegs(record, inputs, labels)


def create_tfrecord_imgsegs(name, inputs, labels):
    """
    creates tf record file.
    :param name: filename
    :param inputs: nifti filename list
    :param labels: label filename list
    :return: None
    """
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


def wrap_bytes(value):
    # converts raw bytes into tf feature which can be packed into tf record
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    cli_interface()
