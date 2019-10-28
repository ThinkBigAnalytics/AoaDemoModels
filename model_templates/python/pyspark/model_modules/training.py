import logging

from pyspark.ml.classification import LogisticRegression
from pyspark import SparkConf
from pyspark.sql import SparkSession
from .util import read_dataset_from_url

logging.getLogger("py4j").setLevel(logging.ERROR)

# spark name and other properties are set by the framework launcher in spark-submit. don't override
spark = SparkSession.builder \
    .config(conf=SparkConf()) \
    .getOrCreate()


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    train = read_dataset_from_url(spark, data_conf["url"])

    lr = LogisticRegression(maxIter=hyperparams["maxIter"],
                            regParam=hyperparams["regParam"],
                            elasticNetParam=hyperparams["elasticNetParam"])

    logging.info("Starting training...")

    lr_model = lr.fit(train)

    # Print the coefficients and intercept for logistic regression
    logging.debug("Coefficients: {}".format(str(lr_model.coefficients)))
    logging.debug("Intercept: {}".format(str(lr_model.intercept)))

    logging.info("Finished training")

    # export model artefacts to models/ folder
    lr_model.write().save(spark.conf.get("spark.aoa.modelPath"))
    logging.info("Saved trained model")

    return lr_model