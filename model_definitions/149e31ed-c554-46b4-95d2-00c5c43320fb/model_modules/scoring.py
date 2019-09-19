import json
import logging

from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from .util import read_dataset_from_url

logging.getLogger("py4j").setLevel(logging.ERROR)

# spark name and other properties are set by the framework launcher in spark-submit. don't override
spark = SparkSession.builder \
    .config(conf=SparkConf()) \
    .getOrCreate()


def score(data_conf, model_conf, **kwargs):
    lr_model = LogisticRegressionModel.load(spark.conf.get("spark.aoa.modelPath"))

    test = read_dataset_from_url(spark, data_conf["url"])
    # we need to ensure that test has an id for comparison (almost all datasets will already have a primary key!!)
    test = test.select("*").withColumn("id", monotonically_increasing_id())

    predictions = lr_model.transform(test).select("id", "rawPrediction", "prediction", "probability")

    predictions.write.mode("overwrite").save(data_conf["predictions"])
    logging.info("Finished scoring")


def evaluate(data_conf, model_conf, **kwargs):

    score(data_conf, model_conf, **kwargs)

    test = read_dataset_from_url(spark, data_conf["url"])

    expected = test.select("*").withColumn("id", monotonically_increasing_id())
    actual = spark.read.load(data_conf["predictions"])

    evaluator = BinaryClassificationEvaluator()
    roc = evaluator.evaluate(expected.join(actual, expected.id == actual.id))
    logging.info('Test Area Under ROC: {}'.format(roc))

    with open("models/evaluation.json", "w+") as f:
        json.dump({'roc': roc}, f)

    logging.info("Finished Evaluation")
