
# Overview
The model is based on the Keras IMDB sentiment analytis using a CNN (see [here](https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py)).

# Training
The [training.oy](./DOCKER_GENERIC_RAW/model_modules/training.py) produces the following artifacts in the artefact repository

    model.h5      (architecture and weights file)


# Evaluation
Evaluation is also performed in [scoring.evluate](./DOCKER_GENERIC_RAW/model_modules/scoring.py) and it returns the following metrics

    acc: <acc>
    

# Scoring 
The [scoring.py](./DOCKER_GENERIC_RAW/model_modules/scoring.py) loads the model and metadata and accepts the word embeddings for prediction. Note that it could accept the text and convert the text to the work embedding but this is typically performed by another service.  


## Sample Request

    curl -X POST -H "Content-Type: application/json" -d "@data.json" http://<service-name>/predict
    
