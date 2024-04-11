"""
Module for dataset preprocessing and augmentation.
"""
import gin
import numpy as np
import tensorflow as tf


# Initialize the default random number generator
default_rng = np.random.default_rng()


@gin.configurable
def preprocess(image, img_height, img_width):
    """
    Dataset preprocessing: Normalizing, reshaping and change image from RGB to YUV color space.

    Parameters:
        image (tf.Tensor): Input image tensor.
        img_height (int): Desired height of the image.
        img_width (int): Desired width of the image.

    Returns:
        tf.Tensor: Processed image in YUV color space.
    """

    # Normalize image: `uint8` of range [0, 255] -> `float32` of range [0, 1].
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Resize image while preserving aspect ratio
    shape = tf.cast(tf.shape(image), tf.float32)
    full_image_height = shape[0]
    full_image_width = shape[1]
    max_ratio = tf.reduce_max([img_height / full_image_height,
                               img_width / full_image_width])
    resize_height = tf.cast(tf.round(full_image_height * max_ratio), tf.int32)
    resize_width = tf.cast(tf.round(full_image_width * max_ratio), tf.int32)
    image = tf.image.resize(image, [resize_height, resize_width])

    # Randomly crop image to the desired image shape
    image = tf.image.random_crop(image, (img_height, img_width, 3))

    # Convert the image to YUV color space
    image_yuv = tf.image.rgb_to_yuv(image)
    return image_yuv


def split_yuv_image(image_yuv):
    """
    Split the YUV image into Y (luma) and UV (chroma) components.

    Parameters:
        image_yuv (tf.Tensor): Input YUV image tensor.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Y (luma) and UV (chroma) components.
    """
    image_y = image_yuv[..., 0:1]  # Keep channel dimension
    image_uv = image_yuv[..., 1:]
    return image_y, image_uv


def is_color_image(image_rgb):
    """
    Check if the input image is in color or grayscale.

    Parameters:
        image_rgb (tf.Tensor): Input RGB image tensor with integer values.

    Returns:
        tf.Tensor: True if the image is RGB, False if grayscale.
    """
    r, g, b = tf.split(image_rgb, num_or_size_splits=3, axis=-1)
    return tf.logical_not(tf.logical_and(tf.reduce_all(r == g), tf.reduce_all(r == b)))


def compute_color_score(image_rgb):
    """
    Compute a color score of an RGB image based on colorfulness of the image.

    Parameters:
        image_rgb (tf.Tensor): Input RGB image tensor with integer values.

    Returns:
        tf.Tensor: Color score indicating the colorfulness of the image
            (higher score -> image is more colorful).

    Implementation from:
    https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
    """
    image_rgb_float = tf.cast(image_rgb, tf.float32)
    r, g, b = tf.split(image_rgb_float, num_or_size_splits=3, axis=-1)
    rg = tf.abs(r - g)
    yb = tf.abs(0.5 * (r + g) - b)
    (rg_mean, rg_std) = (tf.reduce_mean(rg), tf.math.reduce_std(rg))
    (yb_mean, yb_std) = (tf.reduce_mean(yb), tf.math.reduce_std(yb))

    std_root = tf.sqrt(rg_std ** 2 + yb_std ** 2)
    mean_root = tf.sqrt(rg_mean ** 2 + yb_mean ** 2)

    color_score = std_root + (0.3 * mean_root)
    return color_score


def is_colorful(color_score, threshold=15):
    """
    Check if the image is colorful based on the given color score threshold.

    Parameters:
        color_score (tf.Tensor): Color score of the image.
        threshold (float): Threshold value to determine colorfulness.

    Returns:
        tf.Tensor: True if the image is considered colorful, False otherwise.
    """
    return tf.greater_equal(color_score, threshold)


def add_color_score(image_rgb):
    """
    Add color score information to the RGB image.

    Parameters:
        image_rgb (tf.Tensor): Input RGB image tensor.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: RGB image and color score.
    """
    color_score = compute_color_score(image_rgb)
    return image_rgb, color_score


@gin.configurable
def augment(image, random_flip_left_right=True):
    """
    Apply data augmentation to input images.

    Parameters:
        image (tf.Tensor): Input image tensor.
        random_flip_left_right (bool): If the image should be randomly
            flipped horizontally (left to right).

    Returns:
        tf.Tensor: Augmented image.
    """
    if random_flip_left_right:
        image = tf.image.random_flip_left_right(image)

    return image


def generate_user_input(image_y, image_uv, p=1/8., loc_strategy='normal'):
    """
    Generate user input for the given Y (luma) and UV (chroma) components.
    User input is simulated based on the number of points and the position of the points.
    The simulated used input (height, width, 3) consists of the mask (first channel)
    and the corresponding UV value (last two dimensions) for the sampled points.

    Parameters:
        image_y (tf.Tensor): Y (luma) component of the image.
        image_uv (tf.Tensor): UV (chroma) component of the image.
        p (float): Probability of success of geometric distribution which
            samples the number of points.
        loc_strategy (str): The strategy to sample the location of the user inputs.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Y, user input, and UV components.
    """
    assert loc_strategy in ['uniform', 'normal'], \
        f"Invalid sample strategy for the location: {loc_strategy}. Choose 'uniform' or 'normal'."

    # Provide full ground truth color for 1% of the images to encourage the
    # network to copy the colors from the input to the output
    if default_rng.random() < 0.01:
        user_input_mask = tf.ones_like(image_y)
        user_input_uv = image_uv
        try:
            image_user_input = tf.concat([user_input_mask, user_input_uv], axis=-1)
        # Unclear why this error occurs, but this fixed it
        except tf.errors.OutOfRangeError:
            image_user_input = tf.concat([user_input_mask, user_input_uv], axis=-1)
        return image_y, image_user_input, image_uv

    # The number of points is drawn from a geometric distribution
    # (-1, so 0 user points are possible)
    nr_points = default_rng.geometric(p=p) - 1

    # Sample locations
    h, w = image_y.shape[:-1]
    if loc_strategy == 'uniform':
        # Each location is sampled from a 2D uniform distribution as row and column
        samples_row = default_rng.integers(low=0, high=h, size=nr_points)
        samples_column = default_rng.integers(low=0, high=w, size=nr_points)
    elif loc_strategy == 'normal':
        # Each location is sampled from a 2D normal distribution as row and column
        samples_row = np.clip(
            default_rng.normal(loc=h/2.-1, scale=h/4-1, size=nr_points),
            a_min=0, a_max=h-1).astype('int')
        samples_column = np.clip(
            default_rng.normal(loc=w/2.-1, scale=w/4-1, size=nr_points),
            a_min=0, a_max=w-1).astype('int')
    else:
        raise ValueError

    # Input mask is 1.0 where the sampled points are located, else 0.0
    user_input_mask = np.zeros_like(image_y)
    user_input_mask[samples_row, samples_column] = 1.

    # Input UV is the original UV value of the image, where the sampled points are located
    user_input_uv = np.zeros_like(image_uv)
    user_input_uv[samples_row, samples_column] = image_uv.numpy()[samples_row, samples_column]

    try:
        image_user_input = tf.concat([user_input_mask, user_input_uv], axis=-1)
    # Unclear why this error occurs, but this fixed it
    except tf.errors.OutOfRangeError:
        image_user_input = tf.concat([user_input_mask, user_input_uv], axis=-1)
    return image_y, image_user_input, image_uv


# Wrap the NumPy function with tf.py_function
def tf_generate_user_input(image_y, image_uv):
    """
    TensorFlow wrapper for generate_user_input needed for mapping the function.

    Parameters:
        image_y (tf.Tensor): Y (luma) component of the image.
        image_uv (tf.Tensor): UV (chroma) component of the image.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Y (luma), user input, and UV (chroma) components.
    """
    # Use tf.py_function to wrap the NumPy function
    result = tf.py_function(generate_user_input,
                            inp=[image_y, image_uv],
                            Tout=[tf.float32, tf.float32, tf.float32])
    return result
