"""
Module to load and prepare various datasets.
"""

import logging

import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from input_pipeline.preprocessing import (is_color_image, add_color_score, is_colorful,
                                          preprocess, augment, split_yuv_image,
                                          tf_generate_user_input)


@gin.configurable
def load(name, data_dir):
    """
    Loads and prepares dataset based on the provided name.

    Parameters:
        data_dir (str): Directory containing the tensorflow datasets.
        name (str): Name of the dataset.

    Returns:
        Tuple: Prepared training, validation, and test datasets along with
            additional information.
    """
    assert name in ['imagenet_64x64', 'imagenet'], \
        f"Invalid dataset name: {name}. Choose 'imagenet_64x64' or imagenet'."

    if name == "imagenet_64x64":
        logging.info(f"Preparing dataset {name}...")
        # Use validation split as test dataset, because there is no test split
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'imagenet_resized/64x64',
            split=['train[:95%]', 'train[95%:]', 'validation'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

    elif name == "imagenet":
        logging.info(f"Preparing dataset {name}...")
        try:
            (ds_train, ds_val, ds_test), ds_info = tfds.load(
                'imagenet2012',
                split=['train', 'validation[:750]', 'test'],
                shuffle_files=True,
                with_info=True,
                data_dir=data_dir
            )
        # Test dataset is not available on the iss-gpu cluster
        except ValueError:
            logging.info("Test dataset is not available. "
                         "Therefore using validation set as test dataset.")
            (ds_train, ds_val, ds_test), ds_info = tfds.load(
                'imagenet2012',
                split=['train', 'validation[:750]', 'validation'],
                shuffle_files=True,
                with_info=True,
                data_dir=data_dir
            )

    else:
        raise ValueError

    def _preprocess(img_label_dict):
        return img_label_dict['image']

    ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    return prepare(ds_train, ds_val, ds_test, ds_info, caching=False)


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size,
            shuffle_buffer, caching):
    """
    Prepare datasets for training, validation and testing.

    Parameters:
        ds_train (tf.data.Dataset): Training dataset.
        ds_val (tf.data.Dataset): Validation dataset.
        ds_test (tf.data.Dataset): Test dataset.
        ds_info: Additional dataset information.
        batch_size (int): Size of the minibatch.
        shuffle_buffer (int): buffer size for the shuffling of the training dataset.
        caching (bool): Flag indicating whether to use caching.


    Returns:
        Tuple: Prepared training, validation, and test datasets.
    """

    # Prepare training dataset
    ds_train = ds_train.filter(is_color_image)  # Filter out grayscale images
    ds_train = ds_train.map(add_color_score, num_parallel_calls=tf.data.AUTOTUNE)
    # Filter out nearly grayscale images (only for ds_train)
    ds_train = ds_train.filter(lambda image, color_score: is_colorful(color_score))
    ds_train = ds_train.map(lambda image, color_score: (preprocess(image), color_score),
                            num_parallel_calls=tf.data.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(shuffle_buffer)
    ds_train = ds_train.map(lambda image, color_score: (augment(image), color_score),
                            num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.map(lambda image, color_score: (*split_yuv_image(image), color_score),
                            num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.map(lambda image_y, image_uv, color_score:
                            (*tf_generate_user_input(image_y, image_uv), color_score),
                            num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.filter(is_color_image)  # Filter out grayscale images
    ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.map(split_yuv_image, num_parallel_calls=tf.data.AUTOTUNE)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.map(tf_generate_user_input, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.filter(is_color_image)  # Filter out grayscale images
    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(split_yuv_image, num_parallel_calls=tf.data.AUTOTUNE)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.map(tf_generate_user_input, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info
