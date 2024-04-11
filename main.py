import logging
import os
import gin
import gin.tf.external_configurables
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
import wandb
from absl import app, flags

from train import Trainer, cGANTrainer
from evaluation.eval import Tester
from evaluation.interactive_eval import ImageColorizer
from input_pipeline import datasets
from utils import utils_params, utils_misc


# Make AdamW optimizer configurable with gin
gin.external_configurable(AdamW)

# Command line flags
FLAGS = flags.FLAGS

flags.DEFINE_enum('task', required=True, default=None,
                  enum_values=['train', 'eval', 'interactive'],
                  help="Specify whether to train or evaluate a model or use "
                       "the interactive GUI to colorize images.")
flags.DEFINE_boolean('wandb', False, 'Specify whether to enable online wandb logging')
flags.DEFINE_boolean('cgan', False, 'Specify whether to enable pix2pix '
                                    'style cGAN training instead of supervised training. '
                                    'Only used, if "task" is "train".')
flags.DEFINE_string('model_id', '',
                    'Specify id of an existing experiment or create '
                    'an experiment with the specified id.')


def main(argv):

    # Enable mixed precision by default
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # generate folder structures
    run_paths = utils_params.gen_run_folder(FLAGS.model_id)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO, b_stream=True)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    # do not overwrite existing config file
    if not os.path.isfile(run_paths['path_gin']):
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup weights and biases
    wandb_mode = 'disabled'
    if FLAGS.wandb:
        wandb_mode = 'online'
    wandb.init(project='guided-input-colorization', name=run_paths['model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG),
               mode=wandb_mode)

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    if FLAGS.task == 'train':
        if FLAGS.cgan:
            trainer = cGANTrainer(ds_train, ds_val, ds_info, run_paths)
        else:
            trainer = Trainer(ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue

    if FLAGS.task == 'eval':
        # Set loggers to eval
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO, b_stream=True)
        tester = Tester(ds_test, ds_info, run_paths)
        tester.evaluate()

    if FLAGS.task == 'interactive':
        # Set loggers to eval
        utils_misc.set_loggers(run_paths['path_logs_interactive'], logging.INFO, b_stream=True)
        image_colorizer = ImageColorizer(run_paths)
        image_colorizer.run()


if __name__ == "__main__":
    app.run(main)
