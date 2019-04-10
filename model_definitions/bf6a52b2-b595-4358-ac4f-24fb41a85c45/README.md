
# Overview
Simple least squares regression using GBM module from R taken from [here](https://github.com/gbm-developers/gbm/blob/master/demo/gaussian.R)

# Datasets
The dataset used is randomly generated data for demo purposes. Therefore it doesn't expect any data information to be provided.

# Training
The [training.R](DOCKER/model_modules/training.R) produces the following artifacts

- model.rds     (gbm parameters)

# Evaluation
Evaluation is also performed in [scoring.evluate](DOCKER/model_modules/scoring.R) and it returns the following metrics

    accuracy: <acc>

# Scoring 
The [scoring.R](DOCKER/model_modules/scoring.R) is responsible loads the model and metadata and accepts the dataframe for
for prediction. 

# Sample Request

    curl -X POST -H "Content-Type: application/json" -d "@data.json" http://<service-name>/predict
