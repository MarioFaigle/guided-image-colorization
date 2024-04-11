"""
Module for evaluating a guided colorization model on the test dataset.
"""

import logging
import os
import gin
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from models.architectures import unet
from models.utils import load_checkpoint, print_model_summary


matplotlib.use('Agg')


@gin.configurable()
class Tester(object):
    """
    Class for testing a TensorFlow model on a test dataset.
    """
    def __init__(self, ds_test, ds_info, run_paths):
        """
        Initialize a Tester object.

        Parameters:
            ds_test (tf.data.Dataset): The test dataset.
            ds_info: Information about the dataset.
            run_paths (dict): Dictionary containing paths for the experiment run.
        """
        # Log paths for reference
        logging.info(f"Evaluation logs are stored in {run_paths['path_model_id']}")
        logging.info(f"Visualizations are stored in {run_paths['path_vis']}")

        self.ds_test = ds_test
        self.ds_info = ds_info
        self.run_paths = run_paths

        # Set the model and load the latest checkpoint
        self.model = unet()
        print_model_summary(self.model, print_fn=logging.info)
        load_checkpoint(self.model, self.run_paths['path_ckpts_train'])

    @tf.function
    def test_step(self, images_y, images_user_input):
        """
        Perform a single test step on a batch of images.

        Parameters:
            images_y (tf.Tensor): Y (luma) component of the images for testing.
            images_user_input (tf.Tensor): User input for testing.
        """
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        images_uv_pred = self.model([images_y, images_user_input], training=False)
        return images_uv_pred

    def visualize_colorization(self, image_name, image_y, image_user_input,
                               image_uv, image_uv_pred):
        """
        Visualize colorization results and save the visualization.

        Parameters:
            image_name (str): Name of the image.
            image_y (tf.Tensor): Y (luma) component of the image.
            image_user_input (tf.Tensor): User input for colorization.
            image_uv (tf.Tensor): Ground truth UV components of the image.
            image_uv_pred (tf.Tensor): Predicted UV components of the image.
        """
        # Set up subplots for visualization
        fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)

        # Grayscale image
        axs[0, 0].set_title('Grayscale Image\n')
        axs[0, 0].imshow(image_y, cmap='gray', interpolation='none')
        axs[0, 0].axis('off')

        # Ground truth color image
        image_yuv = tf.concat([image_y, image_uv], axis=-1)
        image_rgb = tf.image.yuv_to_rgb(image_yuv)
        image_rgb = tf.clip_by_value(image_rgb, clip_value_min=0.0, clip_value_max=1.0)
        axs[0, 1].set_title('Ground Truth Color Image\n')
        axs[0, 1].imshow(image_rgb, interpolation='none')
        axs[0, 1].axis('off')

        # (Simulated) user input
        user_mask = image_user_input[..., 0:1]
        nr_points = tf.math.count_nonzero(user_mask)
        image_user_y = image_y * user_mask
        image_user_uv = image_user_input[..., 1:]
        image_user_yuv = tf.concat([image_user_y, image_user_uv], axis=-1)
        image_user_rgb = tf.image.yuv_to_rgb(image_user_yuv)
        image_user_rgb = tf.clip_by_value(image_user_rgb, clip_value_min=0.0, clip_value_max=1.0)

        # If all points are given, use the complete rgb image
        if tf.reduce_all(tf.cast(user_mask, tf.bool)):
            image_user_rgb = image_rgb
        else:
            # Colorize neighborhood pixels in the same color
            image_user_rgb = tf.Variable(image_user_rgb)
            height, width = image_user_rgb.shape[:2]
            indices = tf.where(tf.not_equal(image_user_rgb, 0))
            for row, col, _ in indices:
                rgb_value = image_rgb[row, col]
                # Extract neighborhood indices
                row_range = slice(max(0, row - width // 20), min(height, row + width // 20 + 1))
                col_range = slice(max(0, col - height // 20), min(width, col + height // 20 + 1))
                # Assign the neighborhood pixels in the same color
                image_user_rgb[row_range, col].assign(rgb_value)
                image_user_rgb[row, col_range].assign(rgb_value)

        axs[1, 0].set_title(f'(Simulated) User Input Overlay\n'
                            f'{nr_points} points given')
        axs[1, 0].imshow(image_y, cmap='gray', interpolation='none')
        axs[1, 0].imshow(image_user_rgb, alpha=0.85)
        axs[1, 0].axis('off')

        # Predicted color image
        image_yuv_pred = tf.concat([image_y, image_uv_pred], axis=-1)
        image_rgb_pred = tf.image.yuv_to_rgb(image_yuv_pred)
        image_rgb_pred = tf.clip_by_value(image_rgb_pred, clip_value_min=0.0, clip_value_max=1.0)
        axs[1, 1].set_title('Predicted Color Image\n')
        axs[1, 1].imshow(image_rgb_pred, interpolation='none')
        axs[1, 1].axis('off')

        # Adjust layout for better visualization
        fig.tight_layout(pad=0)

        # Save figure
        file_path = os.path.join(self.run_paths['path_vis'], f'vis_{image_name}.png')
        plt.savefig(file_path, bbox_inches='tight', dpi=200)
        plt.close()

    def evaluate(self):
        """
        Evaluate the model on the entire test dataset and save visualizations of colorization.
        """
        logging.info("\n##### Starting evaluation #####\n")

        idx = 1
        for images_y, images_user_input, images_uv in self.ds_test:
            images_uv_pred = self.test_step(images_y, images_user_input)
            for image_y, image_user_input, image_uv, image_uv_pred in zip(
                    images_y, images_user_input, images_uv, images_uv_pred):
                self.visualize_colorization(f"test_{idx:06d}",
                                            image_y, image_user_input, image_uv, image_uv_pred)
                idx += 1

        logging.info('\n##### Finished evaluation #####\n')
