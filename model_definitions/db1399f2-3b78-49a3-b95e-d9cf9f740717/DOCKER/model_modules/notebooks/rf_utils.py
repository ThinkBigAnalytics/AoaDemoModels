from teradataml.context.context import create_context as td_create_context
from teradataml.dataframe.dataframe import DataFrame as td_DataFrame

import datetime

import numpy as np
import pandas as pd

from teradataml.analytics import DecisionForest as td_DecisionForest
from teradataml.analytics import DecisionForestPredict as td_DecisionForestPredict 

from teradataml.dataframe.dataframe import copy_to_sql as td_copy_to_sql



server = "tddb-env-d-268.vantage.demo.intellicloud.teradata.com"
user = "Vantage"
password = "analytics"
eng = td_create_context(host = server, username = user, password = password)
conn = eng.connect()

name_of_table = 'siemens_NO2_prediction'



#no change, logic for slicing is in DataAccess.get_data()
class Interval(object):
    """Some date utility. Would be imported from some other module."""
    _format = "%d-%m-%Y-%H-%M-%S"

    def __init__(self, start: datetime, end: datetime):
        assert start < end
        self.start = start
        self.end = end

    def __str__(self):
        return '{}_TO_{}'.format(self.datetime_2_str(self.start), 
                                 self.datetime_2_str(self.end))

    @classmethod
    def datetime_2_str(cls, x: datetime):
        return x.strftime(cls._format)
    

    
class DataAccess(object):
    """Some utility of getting data from Teradata database.

    This data (self.df) is supposed to be the endpoint of some ETL process that happened earlier.
    The data should not be use-case dependent, but enable a variety of down-stream use-cases.
    """
    def __init__(self):
        
        if(False): # datetime is not saved as a timestamp
            self.df = td_DataFrame.from_table(name_of_table)
        else:
            self.df = td_DataFrame.from_query(    
                """SELECT 
                        data.no2, data.luftdruck, data.luftfeuchtigkeit,data.temperatur,
                        cast(data.datetime AS timestamp(0)) as datetime 
                    FROM 
                        siemens_NO2_prediction as data"""
                )
        ##Timezone: By default, Teradata Database converts all TIME and TIMESTAMP values to UTC prior to storing them (see documentation)

    def get_data(
            self,
            interval: Interval,
            columns=slice(None)):   
        return self.df.loc[
            (self.df.datetime >= interval.start) 
            & (self.df.datetime < interval.end), 
            columns]
    

class RandomForestForecaster(object):
    """Some forecast model.

    The public methods of this class flexibly enable usecases such as model selection via backtesting, training and
        forecasting in operative mode.

    Maybe this is supported by Vantage out-of-the-box by some high-level ML APIs.
    If not, one would have to implement this behavior more low-level on the vantage stack.

    """

    target = 'no2'

    features = [
        'luftdruck',
        'luftfeuchtigkeit',
        'temperatur',
        'yr', # year is keyword for SQL
        'hr', # hour is keyword for SQL
        'calendarweek',
        'mnth',# month is keyword for SQL
    ]

    def __init__(
            self,
            data_access,
            save_dir="",
    ):
        self.data_access = data_access
        self.save_dir = save_dir

    def _add_temporal_features(self, df):
        """Adds some temporal feature columns.

        This would be considered a temporal state. Every use-case may do this in a different way.
        """
        
        # this is a bit clumsy at the moment, so from a tdml point of view, 
        # it would be better do directly create these features when loading with the DataAccess Function.
        
        
        # register df as temporary SQL df to make a query on it, unfortunately, 
        #tdml cannot yet convert timestamp, but we can use the full expressivenes of teradata SQL
        randomsql_name = "temp" + str(np.random.randint(1000000000,9999999999))
        df.to_sql(randomsql_name,if_exists="replace")
        # get temporal features
        df = td_DataFrame.from_query(
        """
                SELECT YEAR(data.datetime) as yr, 
                    HOUR(data.datetime) as hr, 
                    MONTH(data.datetime) as mnth, 
                    td_day_of_week(data.datetime) as dayofweek,
                    WEEKNUMBER_OF_YEAR(data.datetime) as calendarweek,
                    data.*
                FROM 
                """ +randomsql_name + """ 
                     as data; """
        
        
        )
        #TODO: convert to timestamp

        return df

    def _drop_rows_with_nans_in_columns(self, df, columns=None):
        """Drops rows based on some criteria.

        This would be considered a temporal state. Every use-case may do this in a different way.
        """
        #this is identical
        return df.dropna(subset=columns)

### One hot encoding

    def _get_one_hot_features(self, df):
        
        #as it was specified before, this is done only for weekdays
        #tdml does not yet have onehotencoding included, but there is still an easy, scalable solution:
        randomsql_name = "temp" + str(np.random.randint(1000000000,9999999999))
        df.to_sql(randomsql_name,if_exists="replace")

        #1. get unique values of the column        (could be easily made generic)
        columnname = "dayofweek"
        unique_values = td_DataFrame.from_query(
            "SELECT DISTINCT  {}  as {}    FROM  {}".format(columnname,columnname,randomsql_name)
        ).to_pandas()

        unique_values = list(unique_values.iloc[:,0])

        #2 get unity matrix, which will be pushed to the cluster
        names_of_onehotencoded_features = [columnname+"_"+str(x) for x in unique_values]
        unity_matrix = pd.DataFrame(
                            np.eye(len(unique_values)), 
                             columns=names_of_onehotencoded_features)
        unity_matrix[columnname] = unique_values

        # 3 push unity matrix to the cloud and make left join
        onehotenclookup_name = "onehotenclookup_"+columnname
        td_copy_to_sql(unity_matrix, onehotenclookup_name, if_exists="replace")
        onehotenclookup = td_DataFrame(onehotenclookup_name)

        df = df.join(onehotenclookup, 
                     how="left",
                     on=[columnname,columnname],lsuffix="l_",rsuffix="r_")

        return df, names_of_onehotencoded_features

### train

    def train(self, training_interval):
        df = self.data_access.get_data(interval=training_interval)
        df = self._add_temporal_features(df=df)
        df = self._drop_rows_with_nans_in_columns(df=df, columns=self.features + [self.target])

        #no need for that
        #self.encoder = OneHotEncoder(handle_unknown='ignore')  # (just as a showcase)
        #self.encoder.fit(df.dayofweek.values.reshape(-1, 1))

        # changed since function already returns full dataframe with one-hot-encoded featgures
        #df_one_hot = self._get_one_hot_features(df=df)
        #df = pd.concat([df.reset_index(), df_one_hot], axis=1)
        df,names_of_onehotencoded_features = self._get_one_hot_features(df=df)
        
        
        # Decision Forest
        formula = '{} ~ {}'.format(self.target, " + ".join(self.features+names_of_onehotencoded_features))
        self.estimator = td_DecisionForest(
                                            formula=formula,
                                            data=df,                                            
                                            tree_type="regression",
            
                                            #optimal hyperparameter from the exploration phase
                                            ntree=300, #n_estimators
                                            nodesize=1, # min_samples_leaf
                                            max_depth=50,
                                            mtry = 20
                                            )
        
        #self.estimator = RandomForestRegressor()
        #self.estimator.fit(X=df[self.features + list(df_one_hot.columns)].values, y=df[self.target].values)


### forecast

    def forecast(self, forecast_interval):
        df = self.data_access.get_data(interval=forecast_interval)
        df = self._add_temporal_features(df=df)
        df = self._drop_rows_with_nans_in_columns(df=df, columns=self.features + [self.target])

        # changed since function already returns full dataframe with one-hot-encoded featgures
        #df_one_hot = self._get_one_hot_features(df=df)
        #df = pd.concat([df.reset_index(), df_one_hot], axis=1)
        df,names_of_onehotencoded_features = self._get_one_hot_features(df=df)
        
        rfr_predict_out_test = td_DecisionForestPredict(
                                            object=self.estimator,
                                             newdata = df,
                                             id_column = "datetime",
                                             detailed = False,
                                             terms = self.target
                                            )
        return rfr_predict_out_test.result
        

        #df[self.target + '_hat'] = self.estimator.predict(X=df[self.features + list(df_one_hot.columns)].values)
        #df = df.set_index('datetime')[[self.target, self.target + '_hat']].copy()
        #return df

    
    
    def save(self, training_id):
        td_copy_to_sql(self.estimator.predictive_model,"predictivemodel_rf_{}".format(training_id),
                       if_exists='replace')
        
        #joblib.dump(self.estimator, os.path.join(self.save_dir, 'estimator_{}.pkl'.format(training_id)))
        #joblib.dump(self.encoder, os.path.join(self.save_dir, 'encoder_{}.pkl'.format(training_id)))

    def load(self, training_id):
        self.estimator = td_DataFrame("predictivemodel_rf_{}".format(training_id))
#         joblib.load(os.path.join(self.save_dir, 'estimator_{}.pkl'.format(training_id)))
#         self.encoder = joblib.load(os.path.join(self.save_dir, 'encoder_{}.pkl'.format(training_id)))

    def reset(self):
        self.estimator = None
        self.encoder = None

    def get_algorithmic_id(self):
        return 'RandomForestForecaster_v0'
