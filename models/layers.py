"""
Module containing custom layers for the neural network architecture.
"""
import gin
import tensorflow as tf


@gin.configurable
def encoder_block(inputs, filters, kernel_size, padding, strides, activation, batch_norm=True):
    """
    Creates an encoder block used in U-Net architecture.

    Parameters:
        inputs (tf.Tensor): Input tensor or feature map.
        filters (int): Number of filters (output channels) for convolutional layers.
        kernel_size (tuple or int): Size of the convolutional kernel.
            If tuple, specifies height and width.
        padding (str): Type of padding. Supported values: 'valid' or 'same'.
        strides (int or tuple): Stride size for the convolution operation.
        activation (tf.keras.activations): Activation function to apply after each convolution.
        batch_norm (bool): If batch normalization layer should be used within the block.

    Returns:
        (tuple): Tuple containing two tensors - output tensor and skip connection tensor.
    """
    x = inputs

    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               padding=padding,
                               activation=activation)(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               padding=padding,
                               activation=activation)(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    skip = x
    out = tf.keras.layers.Conv2D(filters=filters,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 strides=strides,
                                 activation=activation)(x)

    return out, skip


@gin.configurable
def bottleneck_block(inputs, filters, kernel_size, padding, dilations, activation, batch_norm=True):
    """
    Creates a bottleneck block used in U-Net architecture.

    Parameters:
        inputs (tf.Tensor): Input tensor or feature map.
        filters (int): Number of filters (output channels) for convolutional layers.
        kernel_size (tuple or int): Size of the convolutional kernel.
            If tuple, specifies height and width.
        padding (str): Type of padding. Supported values: 'valid' or 'same'.
        dilations (int or tuple): Dilation size for the convolution operation.
        activation (tf.keras.activations): Activation function to apply after each convolution.
        batch_norm (bool): If batch normalization layer should be used within the block.

    Returns:
        (tf.Tensor): Output tensor.
    """
    x = inputs

    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               padding=padding,
                               dilation_rate=dilations,
                               activation=activation)(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               padding=padding,
                               dilation_rate=dilations,
                               activation=activation)(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               padding=padding,
                               dilation_rate=dilations,
                               activation=activation)(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    return x


@gin.configurable
def decoder_block(inputs, filters, kernel_size, padding, strides, activation,
                  skip_mode='add', batch_norm=True):
    """
    Creates a bottleneck block used in U-Net architecture.

    Parameters:
        inputs (tuple): Tuple containing two tensors - input tensor or feature map and 
            skip connection tensor.
        filters (int): Number of filters (output channels) for convolutional layers.
        kernel_size (tuple or int): Size of the convolutional kernel.
            If tuple, specifies height and width.
        padding (str): Type of padding. Supported values: 'valid' or 'same'.
        strides (int or tuple): Stride size for the convolution operation.
        activation (tf.keras.activations): Activation function to apply after each convolution.
        skip_mode (str): The mode for the skip connection. One of 'add', 'concat'.
            Defaults to 'add'.
        batch_norm (bool): If batch normalization layer should be used within the block.

    Returns:
        (tf.Tensor): Output tensor.
    """
    assert skip_mode in ['add', 'concat'], (f"Invalid skip mode: {skip_mode}. "
                                            f"Choose 'add' or 'concat'.")

    (x, skip) = inputs

    x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        strides=strides,
                                        activation=activation)(x)
    if skip_mode == 'add':
        x = tf.keras.layers.Add()([x, skip])
    else:
        x = tf.keras.layers.Concatenate()([x, skip])
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               padding=padding,
                               activation=activation)(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               padding=padding,
                               activation=activation)(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    return x
