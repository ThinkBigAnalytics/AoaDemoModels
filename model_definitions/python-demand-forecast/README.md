# Overview
A simple regression model to forecast demand of number of orders at different outlets of a meal delivery company. It is a random forest model built using scikit-learn -- the famous python library for machine learning.

A sample notebook is located [here](notebooks/demand_forecasting.ipynb).

# Datasets
The datasets related to this problem can be downloaded from Kaggle available [here](https://www.kaggle.com/kannanaikkal/food-demand-forecasting). You will need to have an account on Kaggle to download the data manually or programmatically through Kaggle API calls. For now, we will assume the data is downloaded as csv files locally. The following Python code can be used to upload the data to Vantage. 

```python
import pandas as pd
from teradataml.dataframe.copy_to import copy_to_sql
from teradataml.dataframe.dataframe import DataFrame

base_info = pd.read_csv('train.csv')
base_info.set_index('id', inplace=True)
copy_to_sql(base_info, table_name='demand_forecast_demo_base', schema_name='AOA_DEMO', if_exists='replace')
```

In addition to the base features included in train.csv, two other datasets are provided related to this problem with additional information on meals, e.g. meal categories, and outlets, e.g. center locations and types. The following code can be used to combine these and create enhanced training datasets.  


```python
meal_info = pd.read_csv('meal_info.csv')
center_info = pd.read_csv('fulfilment_center_info.csv')
copy_to_sql(meal_info, table_name='demand_forecast_demo_meal', schema_name='AOA_DEMO', if_exists='replace')
copy_to_sql(center_info, table_name='demand_forecast_demo_center', schema_name='AOA_DEMO', if_exists='replace')

df = DataFrame.from_query('''
SELECT a.*, b.category, b.cuisine, c.center_type, c.op_area
FROM demand_forecast_demo_base as a
	LEFT JOIN 
	demand_forecast_demo_meal as b 
	ON 
	a.meal_id = b.meal_id
	LEFT JOIN 
	demand_forecast_demo_center as c 
	ON
	a.center_id = c.center_id;
    ''')

n = round(df_train.shape[0]*0.8) #80% data for training
copy_to_sql(df=df.iloc[0:n], table_name="DEMAND_FORECAST_TRAIN", index=False, if_exists="replace")
copy_to_sql(df=df.iloc[n:], table_name="DEMAND_FORECAST_TEST", index=False, if_exists="replace")
```

The above datasets are available in Teradata Vantage and already configured in the demo environment. For reference, the values which are required

Training
```json
{
    "table": "DEMAND_FORECAST_TRAIN"
}
```
Evaluation

```json
{
    "table": "DEMAND_FORECAST_TEST"
}
```

Batch Scoring
```json
 {
     "table": "DEMAND_FORECAST_TEST",
     "predictions": "DEMAND_FORECAST_PREDICTIONS"
 }
 ```


# Training
The [training.py](model_modules/training.py) produces the following artifacts

- model.joblib     (sklearn pipeline with scalers and xgboost model)
- model.pmml       (pmml version of the xgboost model and sklearn pipeline)

# Evaluation
Evaluation is defined in the `evaluate` method in [scoring.py](model_modules/scoring.py) and it returns the following metrics

- R-Squared error
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Mean Squared Log Error (MSLE)

We produce the following plots for each evaluation also

- actual vs predicted target values (number of orders for this problem) for a sample of evaluated examples

We also use shap to save some global explainability plots which help understand the importance and contribution of each feature

- shap feature importance


# Scoring 
This demo modeL supports three types of scoring

 - Batch
 - RESTful
 - In-Vantage (IVSM)

In-Vantage scoring is supported via the PMML model we produce during scoring.

Batch Scoring is supported via the `score` method in [scoring.py](model_modules/scoring.py). As Evaluation is batch score + compare, the scoring logic is already validated with the evaluation step. As batch scoring is run by a scheduler, the scheduler must know how to tell it what dataset (`data_conf`) it should execute on. It does this by using the `scheduler/dataset_template.json` which is templated by the scheduler (airflow in the demo) with things such as dates which are necessary. This will not be necessary anymore after 2.7+ of the AOA as the user will select the dataset template when deploying. 

Again, as this is a demo model where we are reading the dataset from the web, we simply print the scoring results to stdout. A future update of this can use s3.

RESTful scoring is supported via the `ModelScorer` class which implements a predict method which is called by the RESTful Serving Engine. An example request is  

    curl -X POST http://<service-name>/predict \
        -H "Content-Type: application/json" \
        -d '{
            "data": {
                "ndarray": [
                    44957,
                    73,
                    1778,
                    183.33,
                    0,
                    0,
                    "Beverages",
                    "Italian",
                    "TYPE_A",
                    4.0
                ]
            }
        }' 
