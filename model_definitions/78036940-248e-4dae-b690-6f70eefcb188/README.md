# Churn Prediction using Bayes and MLE in SQL

## Dataset Description

    {
    	"hostname": "<vantage-url>",
    	"schema": "<schema>",
    	"model_table": "<schema>.<where to store your model>",
    	"data_table": "<schema>.<your-data-table>",
    	"results_table": "<schema>.<your-evaluation-results>"
    }
    
    
## Training 

Define the [training.sql](model_modules/training.sql).

Note that you can add DROP TABLE statements which will not cause an exception if the table doesn't exist. Instead it will log the failure to drop but continue executing.


## Evaluation

## Scoring