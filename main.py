import argparse

import tensorflow as tf

from inputs import tf_input_training
from model import model_fn


def cli_interface():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    return interface(**vars(args))


def interface(**kwargs):

    test()

    model = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir='./checkpoints0/')

    # enter estimator training
    model.train(input_fn=tf_input_training, max_steps=20)


def test():
    sess = tf.Session()
    x = tf_input_training()

    y = x

    tf.initialize_all_variables().run(session=sess)
    tf.initialize_all_tables().run(session=sess)

    out = sess.run(y)

    f = 1


if __name__ == '__main__':
    cli_interface()
