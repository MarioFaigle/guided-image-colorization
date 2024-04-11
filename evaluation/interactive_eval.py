"""
This module contains a GUI application for interactively colorizing images using a trained model.
"""

import sys
import os
import logging
import gin
import gin.tf.external_configurables
import tensorflow as tf
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QFileDialog, QHBoxLayout,
                             QVBoxLayout, QWidget, QColorDialog, QPushButton)
from PyQt5.QtGui import QPixmap, QImage, QColor

from input_pipeline.preprocessing import split_yuv_image
from models.utils import load_checkpoint, print_model_summary
from models.architectures import unet


class ImageColorizerApp(QMainWindow):
    """
    GUI application for interactively colorizing images using a trained model.
    """
    def __init__(self, model, run_paths):
        """
        Initializes the ImageColorizerApp.

        Parameters:
            model (tf.keras.Model): Trained colorization model.
            run_paths (dict): Dictionary containing paths for the experiment run.
        """
        super().__init__()

        self.image_path = ""
        self.image_name = ""
        self.run_paths = run_paths

        # GUI Images
        self.gui_grayscale_overlay_image = None
        self.gui_colorized_image = None

        self.color_image = None
        self.net_grayscale_image_original_size = None

        # Network Input
        self.net_grayscale_image = None
        self.net_user_input_image = None
        self.net_colorized_image = None

        # Model
        self.model = model

        self.net_height = gin.query_parameter('preprocess.img_height')
        self.net_width = gin.query_parameter('preprocess.img_width')

        self.display_height = None
        self.display_width = None

        self.init_ui()

    def init_ui(self):
        """
        Initializes the user interface for the ImageColorizerApp.
        """
        self.setWindowTitle("Image Colorizer")
        self.setGeometry(100, 100, 800, 400)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Create a horizontal layout for displaying images side by side
        self.image_layout = QHBoxLayout()

        self.grayscale_label = QLabel(self)
        self.image_layout.addWidget(self.grayscale_label)

        self.color_label = QLabel(self)
        self.image_layout.addWidget(self.color_label)

        self.layout.addLayout(self.image_layout)

        # Create a button to load the image
        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_images)
        self.layout.addWidget(self.load_button)

        # Create a button to save the images
        self.save_button = QPushButton("Save Images", self)
        self.save_button.clicked.connect(self.save_images)
        self.layout.addWidget(self.save_button)

    def load_images(self):
        """
        Loads and preprocesses the selected image.
        """
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                         "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if self.image_path:
            self.image_name, ext = os.path.splitext(os.path.basename(self.image_path))

            image_raw = tf.io.read_file(self.image_path)
            image = tf.image.decode_image(image_raw)
            if image.shape[-1] == 4:
                image = image[..., :3]
            elif image.shape[-1] == 1:
                image = tf.image.grayscale_to_rgb(image)

            self.display_height, self.display_width = image.shape[:2]

            # Grayscale image in RGB format to overlay color pixels
            gui_grayscale_image = QImage(image.numpy(), self.display_width, self.display_height,
                                         3 * self.display_width, QImage.Format_RGB888)
            self.gui_grayscale_overlay_image = (gui_grayscale_image
                                                .convertToFormat(QImage.Format_Grayscale8)
                                                .convertToFormat(QImage.Format_RGB888))
            self.color_image = QImage(self.image_path)

            # Load network input
            image_net = tf.cast(image, tf.uint8)
            image_net = tf.image.convert_image_dtype(image_net, tf.float32)
            image_net = tf.image.rgb_to_yuv(image_net)
            image_net_y, image_net_uv = split_yuv_image(image_net)
            self.net_grayscale_image_original_size = image_net_y
            self.net_grayscale_image = tf.expand_dims(tf.image.resize(
                image_net_y, [self.net_height, self.net_width]), axis=0)

            self.net_user_input_image = tf.Variable(tf.zeros(
                self.net_grayscale_image.shape[:-1] + (3,)))
            self.update_color_image()

            self.update_display()

            # Adjust window size based on the loaded image dimensions
            QApplication.instance().processEvents()
            self.adjustSize()
            self.setFixedSize(self.size())  # Set the desired width and height

            logging.info(f"Loaded image {self.image_name}")

    def update_display(self):
        """
        Updates the displayed images in the GUI.
        """
        # Clear previous content of labels
        self.grayscale_label.clear()
        self.color_label.clear()

        # Grayscale Image
        grayscale_pixmap = QPixmap.fromImage(self.gui_grayscale_overlay_image)
        self.grayscale_label.setPixmap(grayscale_pixmap)

        # Colorized Image
        colorized_pixmap = QPixmap.fromImage(self.gui_colorized_image)
        self.color_label.setPixmap(colorized_pixmap)

    def update_color_image(self, selected_color=None, row=None, column=None):
        """
        Updates the grayscale overlay image and the colorized image after the user selected a color.

        Parameters:
            selected_color (tuple): RGB values of the selected color.
            row (int): Row index of the selected pixel.
            column (int): Column index of the selected pixel.
        """
        if selected_color is not None:
            # Update user input
            rgb_color = tf.constant(selected_color[:-1], tf.uint8)
            # User mask
            row_scaled = int(row * self.net_height / self.display_height)
            column_scaled = int(column * self.net_width / self.display_width)
            self.net_user_input_image[0, row_scaled, column_scaled, 0].assign(1.)
            # User uv color
            uv_color = tf.image.rgb_to_yuv(tf.image.convert_image_dtype(rgb_color, tf.float32))[1:]
            self.net_user_input_image[0, row_scaled, column_scaled, 1:].assign(uv_color)

        image_uv_pred = self.model([self.net_grayscale_image,
                                    tf.constant(self.net_user_input_image)],
                                   training=False)
        image_uv_pred = tf.image.resize(image_uv_pred, (self.display_height, self.display_width))
        image_yuv_pred = tf.concat([self.net_grayscale_image_original_size,
                                    image_uv_pred[0]], axis=-1)
        image_rgb_pred = tf.image.yuv_to_rgb(image_yuv_pred)
        image_rgb_pred = tf.clip_by_value(image_rgb_pred, clip_value_min=0.0, clip_value_max=1.0)
        image_rgb_pred = image_rgb_pred * 255
        image_rgb_pred = tf.cast(image_rgb_pred, tf.uint8)
        self.net_colorized_image = image_rgb_pred

        self.gui_colorized_image = QImage(self.net_colorized_image.numpy(),
                                          self.display_width, self.display_height,
                                          3 * self.display_width, QImage.Format_RGB888)

    def mousePressEvent(self, event):
        """
        Handles mouse press events to capture user input for colorization.

        Parameters:
            event: Mouse press event.
        """
        if self.gui_grayscale_overlay_image:
            # Get the position of the mouse relative to the grayscale_label
            pos = self.grayscale_label.mapFrom(self, event.pos())

            # if clicked into grayscale image
            if pos.x() <= self.grayscale_label.width() and pos.y() <= self.grayscale_label.height():
                x = pos.x()
                y = pos.y()

                color_dialog = QColorDialog.getColor()
                if color_dialog.isValid():
                    selected_color = color_dialog.getRgb()
                    self.update_color_image(selected_color, row=y, column=x)
                    self.colorize_pixel(x, y, selected_color)
                    self.update_display()

    def colorize_pixel(self, x, y, color):
        """
        Colorizes the selected pixel and its neighborhood in the grayscale image.

        Parameters:
            x (int): X-coordinate of the selected pixel.
            y (int): Y-coordinate of the selected pixel.
            color (tuple): RGB values of the selected color.
        """
        if self.gui_grayscale_overlay_image:
            # Set pixel color for chosen pixel and neighborhood in for Grayscale Image
            cross_length = min(self.display_width, self.display_height) // 20
            for i in range(x - cross_length, x + cross_length + 1):
                self.gui_grayscale_overlay_image.setPixelColor(i, y, QColor(*color))
            for j in range(y - cross_length, y + cross_length + 1):
                self.gui_grayscale_overlay_image.setPixelColor(x, j, QColor(*color))

    def save_images(self):
        """
        Saves the original, grayscale, user-input, and colorized images to designated paths.
        """
        if (self.net_grayscale_image_original_size is not None
                and self.gui_grayscale_overlay_image is not None
                and self.color_image is not None
                and self.gui_colorized_image is not None):

            # Save the original image
            original_path = os.path.join(self.run_paths["path_vis_interactive"],
                                         f"{self.image_name}_0_original.png")
            self.color_image.save(original_path)

            # Save the original grayscale image
            grayscale_path = os.path.join(self.run_paths["path_vis_interactive"],
                                          f"{self.image_name}_0_grayscale.png")
            plt.imsave(grayscale_path,
                       tf.image.grayscale_to_rgb(self.net_grayscale_image_original_size).numpy())

            # Find out how many times this image was already colorized
            nr_colorized = 1
            while True:
                user_input_path = os.path.join(self.run_paths["path_vis_interactive"],
                                               f"{self.image_name}_{nr_colorized}_user_input.png")
                colorized_path = os.path.join(self.run_paths["path_vis_interactive"],
                                              f"{self.image_name}_{nr_colorized}_colorized.png")
                if os.path.isfile(user_input_path):
                    nr_colorized += 1
                else:
                    break

            # Save the user input image
            self.gui_grayscale_overlay_image.save(user_input_path)

            # Save the colorized image
            self.gui_colorized_image.save(colorized_path)

            logging.info(f"Images saved to:\n"
                         f"Original image: {original_path}\n"
                         f"Grayscale image: {grayscale_path}\n"
                         f"User Input: {user_input_path}\n"
                         f"Colorized: {colorized_path}")
        else:
            logging.warning("Please load an image before saving.")


class ImageColorizer(object):
    """
    Class for running the image colorization process.
    """
    def __init__(self, run_paths):
        """
        Initializes the ImageColorizer.

        Parameters:
            run_paths (dict): Dictionary containing paths for the experiment run.
        """
        # Log paths for reference
        logging.info(f"Interactive logs are stored in {run_paths['path_model_id']}")
        logging.info(f"Visualizations of the image colorizer "
                     f"are stored in {run_paths['path_vis_interactive']}")

        self.run_paths = run_paths

        # Set the model and load the latest checkpoint
        self.model = unet()
        print_model_summary(self.model, print_fn=logging.info)
        load_checkpoint(self.model, self.run_paths['path_ckpts_train'])

    def run(self):
        """
        Runs the image colorization application.
        """
        app = QApplication(sys.argv)
        window = ImageColorizerApp(self.model, self.run_paths)
        window.show()
        sys.exit(app.exec_())
