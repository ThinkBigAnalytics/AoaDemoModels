
# Overview
Simple least squares regression using GBM module from R taken from [here](https://github.com/gbm-developers/gbm/blob/master/demo/gaussian.R)

# Training
The [training.R](./DOCKER_GENERIC_RAW/model_modules/training.R) produces the following artifacts

- model.rds     (gbm parameters)

# Evaluation
Evaluation is also performed in [scoring.evluate](./DOCKER_GENERIC_RAW/model_modules/scoring.R) and it returns the following metrics

    accuracy: <acc>

# Scoring 
The [scoring.R](./DOCKER_GENERIC_RAW/model_modules/scoring.R) is responsible loads the model and metadata and accepts the dataframe for
for prediction. 

# Sample Request

    curl -X POST -H "Content-Type: application/json" -d "@data.json" http://<service-name>/predict
