import logging
import tensorflow as tf


def set_loggers(path_log=None, logging_level=0, b_stream=False, b_debug=False):
    """
    Set up loggers for both standard logging and TensorFlow logging.

    Parameters:
        path_log (str): Path to the log file. If provided,
            logs will be saved to this file (optional).
        logging_level (int): Logging level to be set for both loggers (optional).
        b_stream (bool): If True, logs will also be streamed to the console (optional).
        b_debug (bool): If True, TensorFlow debug mode with log_device_placement
            set to False will be enabled (optional).
    """

    # std. logger
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging_level)

    # tf logger
    logger_tf = tf.get_logger()
    logger_tf.setLevel(logging_level)

    if path_log:
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)
        logger_tf.addHandler(file_handler)

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    if b_debug:
        tf.debugging.set_log_device_placement(False)
