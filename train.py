"""
Module for training a model on the train dataset.
"""

import logging

import gin
import tensorflow as tf
import wandb

from models.utils import load_checkpoint, print_model_summary
from models.architectures import unet, disc


@gin.configurable
class Trainer(object):
    """
    Class for training a TensorFlow model for user-guided image colorization in a supervised manner.
    """
    def __init__(self, ds_train, ds_val, ds_info, run_paths, optimizer, total_steps,
                 log_interval, ckpt_interval, ckpt_path=None):
        """
        Initializes a Trainer object for model training and evaluation.

        Parameters:
            ds_train (tf.data.Dataset): The training dataset.
            ds_val (tf.data.Dataset): The validation dataset.
            ds_info: Information about the dataset.
            run_paths (dict): Paths for different runs.
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to be used for training.
            total_steps (int): Total steps for the training process.
            log_interval (int): Interval for logging training metrics.
            ckpt_interval (int): Interval for saving checkpoints during training.
            ckpt_path (str): Path to load checkpoint if available. (default=None)
        """
        logging.info(f"Checkpoints and logs for training "
                     f"are stored in {run_paths['path_model_id']}")

        # Summary Writer
        self.model = unet()
        print_model_summary(self.model, print_fn=logging.info)

        # Loss objective
        self.loss_object = tf.keras.losses.Huber(0.5)
        # Optimizer needs to be wrapped with LossScaleOptimizer for mixed precision
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(
            self.ckpt, run_paths['path_ckpts_train'], max_to_keep=6)
        if ckpt_path is not None:
            load_checkpoint(self.model, ckpt_path)
        else:
            logging.info("Training starts from scratch.")

        # Metrics
        self.train_weighted_loss = tf.keras.metrics.Mean(name='train_weighted_loss')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

    def log_metrics(self, step):
        """
        Logs the training and validation metrics at a specific step.

        Parameters:
            step (int): The current iteration number.
        """
        template = ('\n### Step {:5d} ###\n'
                    '--- Train ---\n'
                    'Loss: {:.5f} | Weighted Loss: {:.5f}\n'
                    '--- Validation ---\n'
                    'Loss: {:.5f}\n')

        logging.info(template.format(step,
                                     self.train_loss.result(),
                                     self.train_weighted_loss.result(),
                                     self.val_loss.result()))

    def log_wandb(self, step):
        """
        Logs training and validation metrics to Weights & Biases at a specific step.

        Parameters:
            step (int): The current step or iteration number.
        """
        # Log train metrics
        wandb.log({"Loss/Training": self.train_loss.result(),
                   "Weighted Loss/Training": self.train_weighted_loss.result()
                   }, step=step)

        # Log validation metrics
        wandb.log({"Loss/Validation": self.val_loss.result()}, step=step)

    @tf.function
    def train_step(self, images_y, images_user_input, images_uv, color_scores):
        """
        Executes a single training step.

        Parameters:
            images_y (tf.Tensor): Y (luma) component of the images for training.
            images_user_input (tf.Tensor): User input for training.
            images_uv (tf.Tensor): UV (chroma) component of the images for training.
            color_scores (tf.Tensor): Color score for each image in the batch for the
                weighting of the loss.
        """
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            images_uv_pred = self.model([images_y, images_user_input], training=True)
            weighted_loss = (self.loss_object(images_uv, images_uv_pred,
                                              sample_weight=color_scores[..., tf.newaxis])
                             / (tf.reduce_mean(color_scores) + tf.keras.backend.epsilon()))

            # Scale loss because mixed precision used
            scaled_weighted_loss = self.optimizer.get_scaled_loss(weighted_loss)

        # Computed scaled gradients and unscale them because mixed precision used
        scaled_gradients = tape.gradient(scaled_weighted_loss, self.model.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # For logging purpose
        loss = self.loss_object(images_uv, images_uv_pred)

        self.train_weighted_loss.update_state(weighted_loss)
        self.train_loss.update_state(loss)

    @tf.function
    def val_step(self, images_y, images_user_input, images_uv):
        """
        Executes a validation step for the model evaluation.

        Parameters:
            images_y (tf.Tensor): Y (luma) component of the images for validation.
            images_user_input (tf.Tensor): User input for validation.
            images_uv (tf.Tensor): UV (chroma) component of the images for validation.
        """
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        images_uv_pred = self.model([images_y, images_user_input], training=False)
        loss = self.loss_object(images_uv, images_uv_pred)

        self.val_loss.update_state(loss)

    def train(self):
        """
        Initiates the training process for the model.

        Returns:
            val_loss (float): The validation loss after training.
        """
        logging.info("\n##### Starting training #####\n")

        for idx, (images_y, images_user_input, images_uv, color_scores) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images_y, images_user_input, images_uv, color_scores)

            if step % self.log_interval == 0:

                # Reset validation metrics
                self.val_loss.reset_states()

                for val_images_y, val_images_user_input, val_images_uv in self.ds_val:
                    self.val_step(val_images_y, val_images_user_input, val_images_uv)

                # Log metrics
                self.log_metrics(step)
                self.log_wandb(step)

                # Reset train metrics
                self.train_weighted_loss.reset_states()
                self.train_loss.reset_states()

                yield self.val_loss.result().numpy()

            # Save checkpoint
            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                self.manager.save(checkpoint_number=step)

            if step % self.total_steps == 0:
                logging.info(f'\n##### Finished training after {step} steps. #####\n')
                return self.val_loss.result().numpy()


@gin.configurable
class cGANTrainer(object):
    """
    Class for training a TensorFlow model for user-guided image colorization using pix2pix cGAN.
    """
    def __init__(self, ds_train, ds_val, ds_info, run_paths, optimizer_gen, optimizer_disc,
                 lambda_weighted_l1_loss, total_steps, log_interval, ckpt_interval, ckpt_path=None):
        """
        Initializes a cGANTrainer object for model training.

        Parameters:
            ds_train (tf.data.Dataset): The training dataset.
            ds_val (tf.data.Dataset): The validation dataset.
            ds_info: Information about the dataset.
            run_paths (dict): Paths for different runs.
            optimizer_gen (tf.keras.optimizers.Optimizer): The optimizer to be used for training
                the generator.
            optimizer_disc (tf.keras.optimizers.Optimizer): The optimizer to be used for training
                the discriminator.
            lambda_weighted_l1_loss (float | int): The factor for the total
                gan loss =  gan_loss + LAMBDA * weighted_l1_loss.
            total_steps (int): Total steps for the training process.
            log_interval (int): Interval for logging training metrics.
            ckpt_interval (int): Interval for saving checkpoints during training.
            ckpt_path (str): Path to load checkpoint if available. (default=None)
        """
        logging.info(f"Checkpoints and logs for training "
                     f"are stored in {run_paths['path_model_id']}")

        # Summary Writer
        self.generator = unet()
        logging.info("\nGenerator:")
        print_model_summary(self.generator, print_fn=logging.info)
        self.discriminator = disc()
        logging.info("\nDiscriminator:")
        print_model_summary(self.discriminator, print_fn=logging.info)

        # Loss objectives
        self.loss_object_class = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_object_mae = tf.keras.losses.MeanAbsoluteError()
        self.lambda_weighted_l1_loss = lambda_weighted_l1_loss

        # Optimizer needs to be wrapped with LossScaleOptimizer for mixed precision
        self.generator_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_gen)
        self.discriminator_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_disc)

        # Checkpoint Manager
        self.gen_ckpt = tf.train.Checkpoint(model=self.generator)
        self.gen_manager = tf.train.CheckpointManager(
            self.gen_ckpt, run_paths['path_ckpts_train'], max_to_keep=6)
        if ckpt_path is not None:
            load_checkpoint(self.generator, ckpt_path)
        else:
            logging.info("Training starts from scratch.")

        # Metrics
        self.val_gen_l1_loss = tf.keras.metrics.Mean(name='val_gen_l1_loss')
        self.gen_total_loss = tf.keras.metrics.Mean(name='gen_total_loss')
        self.gen_gan_loss = tf.keras.metrics.Mean(name='gen_gan_loss')
        self.gen_weighted_l1_loss = tf.keras.metrics.Mean(name='gen_weighted_l1_loss')
        self.gen_l1_loss = tf.keras.metrics.Mean(name='gen_l1_loss')
        self.disc_loss = tf.keras.metrics.Mean(name='disc_loss')
        self.gen_acc = tf.keras.metrics.BinaryAccuracy(name='gen_acc')
        self.disc_acc_real = tf.keras.metrics.BinaryAccuracy(name='disc_acc_real')
        self.disc_acc_fake = tf.keras.metrics.BinaryAccuracy(name='disc_acc_fake')

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

    def log_metrics(self, step):
        """
        Logs the training and validation metrics at a specific step.

        Parameters:
            step (int): The current iteration number.
        """
        template = ('\n### Step {:5d} ###\n'
                    '--- Train ---\n'
                    'Generator loss: {:.5f} | Generator GAN loss: {:.5f} | '
                    'Generator weighted L1 loss: {:.5f} | Generator L1 loss: {:.5f} | '
                    'Discriminator loss {:.5f}\n'
                    'Generator Accuracy: {:.2%} | '
                    'Discriminator Real Accuracy : {:.2%} | Discriminator Fake Accuracy: {:.2%}\n'
                    '--- Validation ---\n'
                    'Generator L1 loss: {:.5f}\n')

        logging.info(template.format(step,
                                     self.gen_total_loss.result(),
                                     self.gen_gan_loss.result(),
                                     self.gen_weighted_l1_loss.result(),
                                     self.gen_l1_loss.result(),
                                     self.disc_loss.result(),
                                     self.gen_acc.result(),
                                     self.disc_acc_real.result(),
                                     self.disc_acc_fake.result(),
                                     self.val_gen_l1_loss.result()))

    def log_wandb(self, step):
        """
        Logs training and validation metrics to Weights & Biases at a specific step.

        Parameters:
            step (int): The current step or iteration number.
        """
        # Log train metrics
        wandb.log({"Generator/Total Loss/Training": self.gen_total_loss.result(),
                   "Generator/GAN Loss/Training": self.gen_gan_loss.result(),
                   "Generator/Weighted L1 Loss/Training:": self.gen_weighted_l1_loss.result(),
                   "Generator/L1 Loss/Training:": self.gen_l1_loss.result(),
                   "Generator/Accuracy/Training": self.gen_acc.result(),
                   "Discriminator/Disc Loss:": self.disc_loss.result(),
                   "Discriminator/Real Accuracy": self.disc_acc_real.result(),
                   "Discriminator/Fake Accuracy": self.disc_acc_fake.result(),
                   "Generator/L1 Loss/Validation": self.val_gen_l1_loss.result()
                   }, step=step)

    def generator_loss(self, disc_generated_output, gen_images_uv, images_uv, color_scores):
        """
        Computes generator loss.

        Parameters:
            disc_generated_output (tf.Tensor): Output from the discriminator for generated images.
            gen_images_uv (tf.Tensor): Generated UV (chroma) component of the images.
            images_uv (tf.Tensor): Ground truth UV (chroma) component of the images.
            color_scores (tf.Tensor): Color score for each image in the batch for the
                weighting of the loss.

        Returns:
            total_gen_loss (tf.Tensor): Total generator loss.
            gan_loss (tf.Tensor): GAN loss for the generator.
            l1_loss (tf.Tensor): L1 loss for the generator.
        """
        gan_loss = self.loss_object_class(tf.ones_like(disc_generated_output),
                                          disc_generated_output)

        # Weighted mean absolute error (for total loss)
        weighted_l1_loss = (self.loss_object_mae(images_uv, gen_images_uv,
                                                 sample_weight=color_scores[..., tf.newaxis])
                            / (tf.reduce_mean(color_scores) + tf.keras.backend.epsilon()))
        # Mean absolute error (for logging)
        l1_loss = self.loss_object_mae(images_uv, gen_images_uv)

        total_gen_loss = gan_loss + (self.lambda_weighted_l1_loss * weighted_l1_loss)

        return total_gen_loss, gan_loss, weighted_l1_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """
        Computes discriminator loss.

        Parameters:
            disc_real_output (tf.Tensor): Output from the discriminator for real images.
            disc_generated_output (tf.Tensor): Output from the discriminator for generated images.

        Returns:
            total_disc_loss (tf.Tensor): Total discriminator loss.
        """
        real_loss = self.loss_object_class(tf.ones_like(disc_real_output),
                                           disc_real_output)
        generated_loss = self.loss_object_class(tf.zeros_like(disc_generated_output),
                                                disc_generated_output)

        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    @tf.function
    def train_step(self, images_y, images_user_input, images_uv, color_scores):
        """
        Executes a single training step.

        Parameters:
            images_y (tf.Tensor): Y (luma) component of the images for training.
            images_user_input (tf.Tensor): User input for training.
            images_uv (tf.Tensor): UV (chroma) component of the images for training.
            color_scores (tf.Tensor): Color score for each image in the batch for the
                weighting of the loss.
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_images_uv = self.generator([images_y, images_user_input], training=True)

            disc_real_output = self.discriminator([images_y, images_user_input, images_uv],
                                                  training=True)
            disc_generated_output = self.discriminator([images_y, images_user_input, gen_images_uv],
                                                       training=True)

            gen_total_loss, gen_gan_loss, gen_weighted_l1_loss, gen_l1_loss = self.generator_loss(
                disc_generated_output, gen_images_uv, images_uv, color_scores)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

            # Scale loss because mixed precision used
            scaled_gen_total_loss = self.generator_optimizer.get_scaled_loss(gen_total_loss)
            scaled_disc_loss = self.discriminator_optimizer.get_scaled_loss(disc_loss)

        # Computed scaled gradients and unscale them because mixed precision used
        scaled_generator_gradients = gen_tape.gradient(scaled_gen_total_loss,
                                                       self.generator.trainable_variables)
        scaled_discriminator_gradients = disc_tape.gradient(scaled_disc_loss,
                                                            self.discriminator.trainable_variables)
        generator_gradients = self.generator_optimizer.get_unscaled_gradients(
            scaled_generator_gradients)
        discriminator_gradients = self.discriminator_optimizer.get_unscaled_gradients(
            scaled_discriminator_gradients)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

        self.gen_total_loss.update_state(gen_total_loss)
        self.gen_gan_loss.update_state(gen_gan_loss)
        self.gen_weighted_l1_loss.update_state(gen_weighted_l1_loss)
        self.gen_l1_loss.update_state(gen_l1_loss)
        self.disc_loss.update_state(disc_loss)

        self.gen_acc.update_state(tf.ones_like(disc_generated_output),
                                  tf.nn.sigmoid(disc_generated_output))
        self.disc_acc_real.update_state(tf.ones_like(disc_real_output),
                                        tf.nn.sigmoid(disc_real_output))
        self.disc_acc_fake.update_state(tf.zeros_like(disc_generated_output),
                                        tf.nn.sigmoid(disc_generated_output))

    @tf.function
    def val_step(self, images_y, images_user_input, images_uv):
        """
        Executes a validation step for the model evaluation.

        Parameters:
            images_y (tf.Tensor): Y (luma) component of the images for validation.
            images_user_input (tf.Tensor): User input for validation.
            images_uv (tf.Tensor): UV (chroma) component of the images for validation.
        """
        gen_images_uv = self.generator([images_y, images_user_input], training=True)
        gen_l1_loss = self.loss_object_mae(images_uv, gen_images_uv)

        self.val_gen_l1_loss.update_state(gen_l1_loss)

    def train(self):
        """
        Initiates the training process for the model.

        Returns:
            val_loss (float): The validation loss after training.
        """
        logging.info("\n##### Starting training #####\n")

        for idx, (images_y, images_user_input, images_uv, color_scores) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images_y, images_user_input, images_uv, color_scores)

            if step % self.log_interval == 0:

                # Reset validation metrics
                self.val_gen_l1_loss.reset_states()

                for val_images_y, val_images_user_input, val_images_uv in self.ds_val:
                    self.val_step(val_images_y, val_images_user_input, val_images_uv)

                # Log metrics
                self.log_metrics(step)
                self.log_wandb(step)

                # Reset train metrics
                self.gen_total_loss.reset_states()
                self.gen_gan_loss.reset_states()
                self.gen_weighted_l1_loss.reset_states()
                self.gen_l1_loss.reset_states()
                self.disc_loss.reset_states()
                self.gen_acc.reset_states()
                self.disc_acc_real.reset_states()
                self.disc_acc_fake.reset_states()

                yield self.val_gen_l1_loss.result().numpy()

            # Save checkpoint
            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                self.gen_manager.save(checkpoint_number=step)

            if step % self.total_steps == 0:
                logging.info(f'\n##### Finished training after {step} steps. #####\n')
                return self.val_gen_l1_loss.result().numpy()
