# Bank Marketing in R with H2O AutoML
## Overview
Bank Marketing demo model using H2O AutoML in R

## Datasets
The dataset required to train or evaluate this model comes from the UCI Machine Learning repository available [here](https://archive.ics.uci.edu/ml/datasets/bank+marketing).
This dataset is available in Teradata Vantage and already configured in the demo environment.

Training
```json
{
    "table": "<training dataset>"
}
```
Evaluation

```json
{
    "table": "<test dataset>"
}
```

Batch Scoring
```json
 {
     "table": "<score dataset>",
     "predictions": "<output predictions dataset>"
 }
 ```


## Training
The [training.R](model_modules/training.R) produces the following artifacts

- model.h20        (h2o gbm parameters)
- mojo.zip         (h2o mojo version of the trained model)
- h2o-genmodel.jar (jar file with some libraries required for the mojo file when deployed in production)

## Evaluation
Evaluation is performed in [evaluation.R](model_modules/evaluation.R) by the function `evaluate` and it returns the following metrics

    Accuracy: <acc>
    Kappa: <kappa>
    AccuracyLower: <acc_low>
    AccuracyUpper: <acc_upp>
    AccuracyNull: <acc_null>
    AccuracyPValue: <acc_pval>
    McnemarPValue: <mcn_pval>

## Scoring
The [scoring.R](model_modules/scoring.R) loads the model and metadata and accepts the dataframe for prediction.

### Batch mode
In this example, the values to score are in the table 'bank_marketing_data' at Teradata Vantage. The results are saved in the table 'bank_marketing_data_predictions'. When batch deploying, this custom values should be specified:

   | key | value |
   |----------|-------------|
   | table | bank_marketing_data |
   | predictions | bank_marketing_data_predictions |

### RESTful Sample Request

    curl -X POST http://localhost:5000/predict \
            -H "Content-Type: application/json" \
            -d '{
                "data": {
                    "ndarray": [[
                            35,
                            "blue-collar",
                            "married",
                            "primary",
                            "no",
                            5883,
                            "yes",
                            "yes"
                    ]],
                    "names":[
                        "age", 
                        "job", 
                        "marital", 
                        "education", 
                        "default", 
                        "balance", 
                        "housing", 
                        "loan"
                    ]
                }
            }' 
