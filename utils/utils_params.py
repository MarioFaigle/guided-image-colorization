import os
import datetime


def gen_run_folder(model_id=''):
    """
    Create the necessary directory structure and files for organizing an experiment run.

    Parameters:
        model_id (str): Identifier for the experiment.
            If provided, the function checks if an experiment under this ID
            already exists. If it does, the existing paths are returned;
            otherwise, new paths are created based on the timestamp combined
            with the provided ID.

    Returns:
        dict: A dictionary containing the following paths:
            - 'path_model_id': Root path for the experiment run.
            - 'path_logs_train': Path for training logs.
            - 'path_logs_eval': Path for evaluation logs.
            - 'path_ckpts_train': Path for saving training checkpoints.
            - 'path_gin': Path for the GIN configuration file.
    """
    run_paths = dict()

    path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                                                   'experiments', 'guided_image_colorization'))
    if not model_id or not os.path.isdir(os.path.join(path_model_root, model_id)):
        date_creation = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
        run_id = 'run_' + date_creation
        if model_id:
            run_id += '_' + model_id
        run_paths['path_model_id'] = os.path.join(path_model_root, run_id)
    else:
        run_paths['path_model_id'] = os.path.join(path_model_root, model_id)
    run_paths['model_id'] = run_paths['path_model_id'].split(os.sep)[-1]

    run_paths['path_logs_train'] = os.path.join(run_paths['path_model_id'], 'logs', 'train.log')
    run_paths['path_logs_eval'] = os.path.join(run_paths['path_model_id'], 'logs', 'eval.log')
    run_paths['path_logs_interactive'] = os.path.join(run_paths['path_model_id'],
                                                      'logs', 'interactive.log')
    run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts')
    run_paths['path_vis'] = os.path.join(run_paths['path_model_id'], 'vis')
    run_paths['path_vis_interactive'] = os.path.join(run_paths['path_model_id'], 'image_colorizer')
    run_paths['path_gin'] = os.path.join(run_paths['path_model_id'], 'config_operative.gin')

    # Create folders
    for k, v in run_paths.items():
        if any([x in k for x in ['path_model', 'path_ckpts', 'path_vis']]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)

    # Create files
    for k, v in run_paths.items():
        if any([x in k for x in ['path_logs']]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, 'a'):
                    pass  # atm file creation is sufficient

    return run_paths


def save_config(path_gin, config):
    with open(path_gin, 'w') as f_config:
        f_config.write(config)


# https://github.com/google/gin-config/issues/154
def gin_config_to_readable_dictionary(gin_config: dict):
    """
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B

    Parameters:
        gin_config: the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG

    Returns:
        dict: The parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data["/".join([name, k])] = v

    return data
