
# Overview
The model is based on the Keras IMDB sentiment analytis using a CNN (see [here](https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py)).

# Datasets
The dataset used is the IMDB sentiment analysis dataset that is freely available. As this is a demo model, it downloads the data automatically when running either training or evaluation. Therefore it doesn't expect any data information to be provided but we can't provide an empty dataset metadata value so add this. 


# Training
The [training.py](DOCKER/model_modules/training.py) produces the following artifacts in the artefact repository

    model.h5      (architecture and weights file)


# Evaluation
Evaluation is also performed in [scoring.evluate](DOCKER/model_modules/scoring.py) and it returns the following metrics

    acc: <acc>
    loss: <loss>
    

# Scoring 
The [scoring.py](DOCKER/model_modules/scoring.py) loads the model and metadata and accepts the word embeddings for prediction. Note that it could accept the text and convert the text to the work embedding but this is typically performed by another service.  


## Sample Request

    curl -X POST -H "Content-Type: application/json" -d "@data.json" http://<service-name>/predict
    
