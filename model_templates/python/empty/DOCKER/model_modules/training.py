

def train(data_conf, model_conf, **kwargs):
    """Python train method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    hyperparams = model_conf["hyperParameters"]

    # load data & engineer

    print("Starting training...")

    # fit model to training data

    print("Finished training")

    # export model artefacts to models/ folder

    print("Saved trained model")