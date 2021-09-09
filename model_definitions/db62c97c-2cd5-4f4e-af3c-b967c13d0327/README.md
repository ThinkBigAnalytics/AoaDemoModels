# Overview
This is a demo regression model built using a Teradata native Vantage Analytics Library (VAL). The model is applied to a demand forecasting problem. Details of the given problem and a companion scikit-learn-based regression model are available [here](model_definitions/b1e18b12-5ccd-4c94-b96b-c2cebf230150/README.md).

A sample notebook that demonstrates how to build models using VAL is available [here](notebooks/demand_forecasting_with_VAL.ipynb).

# Datasets

The datasets related to this problem can be downloaded from Kaggle available [here](https://www.kaggle.com/kannanaikkal/food-demand-forecasting). You will need to have an account on Kaggle to download the data manually or programmatically through Kaggle API calls. For now, we will assume the data is downloaded as csv files locally. The following Python code can be used to upload the data to Vantage. 

```python
import pandas as pd
from teradataml.dataframe.copy_to import copy_to_sql
from teradataml.dataframe.dataframe import DataFrame

base_info = pd.read_csv('train.csv')
base_info.set_index('id', inplace=True)
copy_to_sql(base_info, table_name='demand_forecast_demo_base', schema_name='AOA_DEMO', if_exists='replace', 
            index=True, index_label='index', primary_index='index')
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

n = round(df_combined.shape[0]*0.8) #80% data for training
copy_to_sql(df = df_combined.iloc[0:n], table_name="DEMAND_FORECAST_TRAIN_VAL", schema_name="AOA_DEMO", if_exists="replace", 
            index=True, index_label="index", primary_index="index")
copy_to_sql(df = df_combined.iloc[n:], table_name="DEMAND_FORECAST_TEST_VAL", schema_name="AOA_DEMO", if_exists="replace", 
            index=True, index_label="index", primary_index="index")
```

The above datasets are available in Teradata Vantage and already configured in the demo environment. For reference, the values which are required

Training
```json
{
    "data_table": "DEMAND_FORECAST_TRAIN_VAL"
}
```
Evaluation

```json
{
    "data_table": "DEMAND_FORECAST_TEST_VAL"
}
```

Batch Scoring
```json
 {
     "data_table": "DEMAND_FORECAST_TEST_VAL",
     "results_table": "DEMAND_FORECAST_TEST_VAL_PREDICTIONS"
 }
 ```


# Training
The VAL model artefacts are stored as Vantage tables ([training.py](model_modules/training.py)). The following two tables are generated as part of training:

- model.model     				(LinReg trained model table)
- model.statistical_measures	(model performance metrics generated as part of model training)

# Evaluation
Evaluation is defined in the `evaluate` method in [scoring.py](model_modules/scoring.py) and it returns the following metrics

- R-Squared error
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

We produce the following plots for each evaluation also

- actual vs predicted target values (number of orders for this problem) for a sample of evaluated examples


# Scoring 
This demo model supports only Batch mode scoring:

 - Batch

Batch Scoring is supported via the `score` method in [scoring.py](model_modules/scoring.py). As Evaluation is batch score + compare, the scoring logic is already validated with the evaluation step. As batch scoring is run by a scheduler, the scheduler must know how to tell it what dataset (`data_conf`) it should execute on. It does this by using the `scheduler/dataset_template.json` which is templated by the scheduler (airflow in the demo) with things such as dates which are necessary. This will not be necessary anymore after 2.7+ of the AOA as the user will select the dataset template when deploying. 

Again, as this is a demo model where we are reading the dataset from the web, we simply print the scoring results to stdout. A future update of this can use s3.