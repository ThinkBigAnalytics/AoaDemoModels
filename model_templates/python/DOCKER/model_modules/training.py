

def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperparameters"]

    # load data & engineer

    print("Starting training...")

    # fit model to training data

    print("Finished training")

    # export model artefacts to models/ folder

    print("Saved trained model")