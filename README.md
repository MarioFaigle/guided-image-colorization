# User Guided Image Colorization

This project was part of the Deep Learning Lab at the University of Stuttgart.

The primary objective of image colorization is to produce realistic colored images for historical pictures where 
no ground truth exists. However, leaving the coloring decision entirely to a neural network often results in the 
selection of a possible but incorrect color. This is particularly noticeable in the eye color of famous individuals. 
In this project, we address this issue by incorporating additional user input, allowing users to select individual 
pixels and color them according to their preferences. The user can conveniently input their desired changes through 
the UI, and only a few pixels are needed to generate realistic results. In addition to correcting wrongly colored 
historical images, the user can also modify any colored image to appear in different colors.

## Table of contents
1. [Training the model](#training-the-model)
2. [Using the user interface](#using-the-user-interface)
3. [Results](#results)
4. [Weights & Biases](#weights--biases)
5. [References](#references)
6. [Team](#team)

## Training the model

### Getting Started

To train or evaluate a model run:
```bash
python3 main.py --task <state> --model_id <your_experiment_name>
```

The following subsections explain how to configure training and evaluation.
Adjust the configuration parameters in [`config.gin`](configs/config.gin) 
for customized training and evaluation settings.

### Training

Key parameters for training configuration include:
* `load.name`: Dataset selection (imagenet or imagenet_64x64 for full resolution or tiny dataset).
* `preprocess.img_height`, `preprocess.img_width`: Customize image resolution to be trained on.
* `unet.base_filters`, `unet.depth`, `unet.width`: Customize model architecture to be trained on.
* `Trainer.ckpt_path`: Continuation of training from a checkpoint.

For supervised training adjust the supervised related configuration parameters.
Train a supervised model with
```bash
python3 main.py --task train --model_id <your_experiment_name>
```

For cGAN training adjust the cGAN related configuration parameters.
Train a cGAN model with
```bash
python3 main.py --task train --cgan --model_id <your_experiment_name>
```

### Evaluation

Important evaluation parameters are:
* `load.name`: Dataset selection (imagenet for full resolution or imagenet_64x64 for tiny dataset).
* `preprocess.img_height`, `preprocess.img_width`: Customize image resolution to be trained on.
* `unet.base_filters`, `unet.depth`, `unet.width`: Customize model architecture to be trained on.

Evaluate a trained model from an experiment called *model_id* with:
```bash
python3 main.py --task eval --model_id <model_id>
```

## Using the user interface

Run the user interface with a trained model from an experiment called *model_id* with:
```bash
python3 main.py --task interactive --model_id <model_id>
```
Loading and saving images in the GUI can be done by clicking the respective buttons. Choosing a color can be done by clicking into the grayscale image. The colorization output is updated in real time.
![GUI with parrot](/imgs/parrot-gif.gif)

## Results

With enough training, very good results can be achieved with just a few user inputs. The smaller the trained image size, the harder it is for the network to distinguish between fine-grained objects. In our experiments, supervised and cGAN training produced similar results, only with cGAN being slightly better at choosing a color where no user input was given.

Some example colorizations can be seen here:
![Example images](/imgs/example-imgs.png)

## Weights & Biases

To log your results to Weights & Biases during training and evaluation simply add 
the flag `--wandb` to the previous commands.

Example of  a run logged with Weights & Biases: 
![Weights and biases logging](/imgs/gic_wandb.png)

## References
[1] Real-Time User-Guided Image Colorization with Learned Deep Priors (https://arxiv.org/abs/1705.02999)

## Team
- Mario Faigle
- Paul Wiench