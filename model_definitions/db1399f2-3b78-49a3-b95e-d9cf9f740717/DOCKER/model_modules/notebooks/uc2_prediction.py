from rf_utils import *

from teradataml.context.context import create_context as td_create_context
from teradataml.dataframe.dataframe import copy_to_sql as td_copy_to_sql

server = "tddb-env-d-268.vantage.demo.intellicloud.teradata.com"
user = "Vantage"
password = "analytics"
eng = td_create_context(host = server, username = user, password = password)
conn = eng.connect()

# check whether we can include a function that pushes results to the cloud, instead of having to establish here another connection to vantage



def run_usecase_prediction():
    """Load a model by id, predict forecast_interval and save predictions for later flexible usage."""
    
    training_interval = Interval(start=datetime.datetime(year=2017, month=1, day=1),
                                 end=datetime.datetime(year=2017, month=2, day=1))
    forecast_interval = Interval(start=datetime.datetime(year=2018, month=6, day=2),
                                 end=datetime.datetime(year=2018, month=6, day=9))
    model = RandomForestForecaster(data_access=DataAccess())
    # string conversion
    str_int = str(training_interval).replace("-","_")
    
    model.load(training_id='{}_{}'.format(model.get_algorithmic_id(), str_int))
    
    results = model.forecast(forecast_interval=forecast_interval)
    str_int_forc = str(forecast_interval).replace("-","_")
    
    df = model.forecast(forecast_interval=forecast_interval)
    # results are also being pushed to the cloud
    
    ########
    #### this should be done through a RandomForestForecaster Function!
    td_copy_to_sql(df,
                  '{}_{}_{}'.format(model.get_algorithmic_id(), str_int, str_int_forc),if_exists="replace")
                  

    

if __name__ == '__main__':
    run_usecase_prediction()