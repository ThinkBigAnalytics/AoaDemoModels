import json
import logging

from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark import SparkConf
from pyspark.sql import SparkSession

logging.getLogger("py4j").setLevel(logging.ERROR)

# spark name and other properties are set by the framework launcher in spark-submit. don't override
spark = SparkSession.builder \
    .config(conf=SparkConf()) \
    .getOrCreate()


def evaluate(data_conf, model_conf, **kwargs):
    lr_model = LogisticRegressionModel.load(spark.sparkContext._conf.get("spark.ammf.model.path", "file:///tmp/lr"))
    test = spark.read.format("libsvm").load(data_conf["data_path"])
    predictions = lr_model.transform(test)
    
    evaluator = BinaryClassificationEvaluator()
    roc = evaluator.evaluate(predictions)
    print('Test Area Under ROC: {}'.format(roc))
    
    with open("models/evaluation.json", "w+") as f:
        json.dump({'roc': roc}, f)
