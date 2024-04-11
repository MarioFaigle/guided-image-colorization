"""
Module containing different neural network architectures.
"""


import gin
import numpy as np
import tensorflow as tf

from models.layers import encoder_block, bottleneck_block, decoder_block


@gin.configurable
def unet(base_filters, depth, width, kernel_size=(3, 3), padding='same', strides=2,
         dilations=2, activation=tf.nn.relu):
    """
    Constructs a U-Net model with customizable depth and width.

    Parameters:
        base_filters (int): Number of base filters for the convolutional layers.
        depth (int): Depth of the U-Net architecture. If set to 0,
            the depth is automatically calculated based on the input size.
        width (int): Width of the U-Net architecture.
            If set lower than 2*depth+1, defaults to 2*depth+1.
        kernel_size (tuple, optional): Size of the convolutional kernel. Defaults to (3, 3).
        padding (str, optional): Padding mode for convolutional layers. Defaults to 'same'.
        strides (int, optional): Stride for downsampling layers in encoder blocks. Defaults to 2.
        dilations (int, optional): Dilation rate for downsampling layers in bottleneck blocks. 
            Defaults to 2.
        activation (function, optional): Activation function. Defaults to tf.nn.leaky_relu.

    Returns:
        (tf.keras.Model): U-Net model.
    """
    # Store input shape
    img_height = gin.query_parameter("preprocess.img_height")
    img_width = gin.query_parameter("preprocess.img_width")

    # automatic depth calculation
    if depth == 0:
        min_size = 8
        input_size = img_height
        depth = int(np.log(input_size/min_size)/np.log(strides))

    residuals = []
    inputs_y = tf.keras.Input((img_height, img_width, 1))
    inputs_user = tf.keras.Input((img_height, img_width, 3))
    x = tf.keras.layers.Concatenate()([inputs_y, inputs_user])

    # Encoder
    for i in range(depth):
        x, skip = encoder_block(x,
                                filters=base_filters*2**i,
                                kernel_size=kernel_size,
                                padding=padding,
                                strides=strides,
                                activation=activation)
        residuals.append(skip)

    # Bottleneck (if width < 2*depth then if defaults to one bottleneck block)
    x = bottleneck_block(x,
                         filters=base_filters*2**depth,
                         kernel_size=kernel_size,
                         padding=padding,
                         dilations=dilations,
                         activation=activation)
    for i in range((width-1)-2*depth):
        x = bottleneck_block(x,
                             filters=base_filters*2**depth,
                             kernel_size=kernel_size,
                             padding=padding,
                             dilations=dilations,
                             activation=activation)

    # Decoder
    for i in range(depth):
        skip = residuals[(depth-1)-i]
        x = decoder_block((x, skip),
                          filters=base_filters*2**((depth-1)-i),
                          kernel_size=kernel_size,
                          padding=padding,
                          strides=strides,
                          activation=activation)

    # Output layer
    x = tf.keras.layers.Conv2D(filters=2,
                               kernel_size=kernel_size,
                               padding=padding)(x)
    outputs = tf.keras.layers.Activation('tanh', dtype='float32')(x)

    return tf.keras.Model(inputs=[inputs_y, inputs_user], outputs=outputs, name='unet')


@gin.configurable
def disc(base_filters, depth=0, kernel_size=(3, 3), padding='same', strides=2,
         activation=tf.nn.relu):
    """
    Constructs a discriminator model with customizable depth.

    Parameters:
        base_filters (int): Number of base filters for the convolutional layers.
        depth (int, optional): Depth of the discriminator architecture.
            If set to 0, the depth is automatically calculated based on the input size.
            Defaults to 0.
        kernel_size (tuple, optional): Size of the convolutional kernel. Defaults to (3, 3).
        padding (str, optional): Padding mode for convolutional layers. Defaults to 'same'.
        strides (int, optional): Stride for downsampling layers in encoder blocks. Defaults to 2.
        activation (function, optional): Activation function. Defaults to tf.nn.relu.

    Returns:
        (tf.keras.Model): Discriminator model.
    """

    # Store input shape
    img_height = gin.query_parameter("preprocess.img_height")
    img_width = gin.query_parameter("preprocess.img_width")

    # automatic depth calculation
    if depth == 0:
        min_size = 1
        input_size = img_height
        depth = int(np.log(input_size / min_size) / np.log(strides))

    inputs_y = tf.keras.Input((img_height, img_width, 1))
    inputs_user = tf.keras.Input((img_height, img_width, 3))
    inputs_uv = tf.keras.Input((img_height, img_width, 2))
    x = tf.keras.layers.Concatenate()([inputs_y, inputs_user, inputs_uv])

    for i in range(depth):
        x, _ = encoder_block(x,
                             filters=base_filters*2**i,
                             kernel_size=kernel_size,
                             padding=padding,
                             strides=strides,
                             activation=activation)
        x = tf.keras.layers.Dropout(0.2)(x)

    # Output layer
    x = tf.keras.layers.Conv2D(filters=1,
                               kernel_size=kernel_size,
                               padding=padding)(x)
    outputs = tf.keras.layers.Activation('linear', dtype='float32')(x)

    return tf.keras.Model(inputs=[inputs_y, inputs_user, inputs_uv], outputs=outputs, name='disc')
