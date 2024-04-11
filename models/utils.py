"""
Utilities for tf.keras.Model.
"""

import sys
import logging
import tensorflow as tf


def load_checkpoint(model, ckpt_path):
    """
    Load the latest checkpoint for the model for the specified experiment.
    If this is not possible, log an error and exit the program.
    Note: The model needs to be built.

    Parameters:
        model (tf.keras.Model): The Keras model for which to load the checkpoint.
        ckpt_path (str): The path to the directory containing checkpoints for the experiment.

    Raises:
        SystemExit: If loading the checkpoint encounters an error,
            log an error message and exit the program.
    """
    ckpt = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
        restore_status = ckpt.restore(latest_ckpt)
        try:
            restore_status.expect_partial()
            restore_status.assert_consumed()
        except AssertionError:
            logging.error("The model architecture and the checkpoint parameters do not match.")
            sys.exit(1)
        logging.info(f"Restored checkpoint from {latest_ckpt}.")
    else:
        logging.error(f"No checkpoint found at {ckpt_path}.")
        sys.exit(1)


def print_model_summary(model, print_fn=None):
    """Prints a string summary of the network.

    Parameters:
        model (tf.keras.Model): The Keras model to summarize. Model has to be built.
        print_fn (function, optional): Function used to print the summary.
            If None, defaults to standard print function.
    """
    model.summary(print_fn=print_fn)
