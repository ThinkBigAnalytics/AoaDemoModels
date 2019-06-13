import json

from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def evaluate(spark, data_conf, model_conf, **kwargs):
    lr_model = LogisticRegressionModel.load(spark.sparkContext._conf.get("spark.ammf.model.path", "file:///tmp/lr"))
    test = spark.read.format("libsvm").load(data_conf["data_path"])
    predictions = lr_model.transform(test)
    
    evaluator = BinaryClassificationEvaluator()
    roc = evaluator.evaluate(predictions)
    print('Test Area Under ROC: {}'.format(roc))
    
    with open("models/evaluation.json", "w+") as f:
        json.dump({'roc': roc}, f)
