import json
import logging
import urllib.request

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
    lr_model = LogisticRegressionModel.load(data_conf["model_path"])

    # for this demo we're downloading the dataset locally and then reading it. This is obviously not production setting
    # https://raw.githubusercontent.com/apache/spark/branch-2.4/data/mllib/sample_libsvm_data.txt
    urllib.request.urlretrieve(data_conf["url"], "/tmp/data.txt")

    test = spark.read.format("libsvm").load("/tmp/data.txt")

    predictions = lr_model.transform(test)

    evaluator = BinaryClassificationEvaluator()
    roc = evaluator.evaluate(predictions)
    print('Test Area Under ROC: {}'.format(roc))

    with open("models/evaluation.json", "w+") as f:
        json.dump({'roc': roc}, f)
