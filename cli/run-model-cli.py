#!/usr/bin/env python

import json
import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Run model training or scoring locally')
    parser.add_argument('model_id', type=str, help='Which model_id to use')
    parser.add_argument('mode', type=str, help='The model (train or evaluate)')
    parser.add_argument('-d', '--data', type=str, help='Yaml file containing data configuration')

    args = parser.parse_args()

    base_path = os.path.dirname(os.path.realpath(__file__)) + "/../"

    model_dir = base_path + "./model_definitions/" + args.model_id

    with open(model_dir + "/model.json", 'r') as f:
        model_definition = json.load(f)

    with open(model_dir + "/DOCKER/config.json", 'r') as f:
        model_conf = json.load(f)

    if args.data:
        with open(args.data, 'r') as f:
            data_conf = json.load(f)

    else:
        logging.info("Using empty dataset definition")
        data_conf = {}

    if model_definition["language"] == "python":
        sys.path.append(model_dir + "/DOCKER")
        import model_modules

        if args.mode == "train":
            model_modules.training.train(data_conf, model_conf)
        elif args.mode == "evaluate":
            model_modules.scoring.evaluate(data_conf, model_conf)
        else:
            raise Exception("Unsupported mode used: " + args.mode)

    elif model_definition["language"] == "pyspark":
        from pyspark import SparkContext, SparkConf
        from pyspark.sql import SparkSession

        spark = SparkSession.builder \
            .appName("spark-model-runner-cli") \
            .config(conf=SparkConf()) \
            .getOrCreate()

        sys.path.append(model_dir + "/DOCKER")
        import model_modules

        if args.mode == "train":
            model_modules.training.train(spark, data_conf, model_conf)
        elif args.mode == "evaluate":
            model_modules.scoring.evaluate(spark, data_conf, model_conf)
        else:
            raise Exception("Unsupported mode used: " + args.mode)

    else:
        raise Exception("Unsupported cli language: {}".format(model_definition["language"]))


if __name__ == "__main__":
    main()
