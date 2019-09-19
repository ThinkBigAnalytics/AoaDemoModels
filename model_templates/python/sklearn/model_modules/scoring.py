import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
import json

def evaluate(data_conf, model_conf, **kwargs):
    """Python evaluate method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    predict_df = pd.read_csv(data_conf['location'])
    features = 'sepallength,sepalwidth,petallength,petalwidth'.split(',')
    X_predict = predict_df.loc[:, features]
    y_test = predict_df['class']
    knn = joblib.load('models/iris_knn.joblib')

    y_predict = knn.predict(X_predict)
    scores = {}
    scores['accuracy'] = metrics.accuracy_score(y_test, y_predict)
    print("model accuracy is ", scores['accuracy'])

    # dump results as json file evaluation.json to models/ folder
    with open("models/evaluation.json", "w+") as f:
        json.dump(scores, f)
    print("Evaluation complete...")