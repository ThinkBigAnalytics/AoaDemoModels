from rf_utils import *


def run_usecase_model_training():
    """Train and save a model for later flexible usage."""
    #no changes needed, apart from string conversion and removal of time zone
    training_interval = Interval(start=datetime.datetime(year=2017, month=1, day=1),
                                 end=datetime.datetime(year=2017, month=2, day=1))
    model = RandomForestForecaster(data_access=DataAccess())
    model.train(training_interval=training_interval)
    
    str_int = str(training_interval).replace("-","_")
    model.save(training_id='{}_{}'.format(model.get_algorithmic_id(), str_int))
    

if __name__ == '__main__':
    run_usecase_model_training()