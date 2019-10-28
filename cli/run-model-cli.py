#!/usr/bin/env python

# https://github.com/ThinkBigAnalytics/AoaCoreService/issues/78
# Code to be moved into centralised aoa cli

import json
import logging
import sys
import os
import shutil
from jinja2 import Template

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Run model training or scoring locally')
    parser.add_argument('model_id', type=str, help='Which model_id to use')
    parser.add_argument('mode', type=str, help='The model (train or evaluate)')
    parser.add_argument('-d', '--data', type=str, help='Json file containing data configuration')

    args = parser.parse_args()

    base_path = os.path.dirname(os.path.realpath(__file__)) + "/../"

    model_dir = base_path + "./model_definitions/" + args.model_id

    with open(model_dir + "/model.json", 'r') as f:
        model_definition = json.load(f)

    with open(model_dir + "/config.json", 'r') as f:
        model_conf = json.load(f)

    if not args.data:
        raise Exception("Dataset metadata must be specified.")

    with open(args.data, 'r') as f:
        data_conf = json.load(f)

    if os.path.exists('models'):
        logging.info("Cleaning directory {} to store test model artefacts".format(os.getcwd() + "/models"))
        shutil.rmtree("path_to_dir")

    else:
        logging.info("Creating directory {} to store test model artefacts".format(os.getcwd() + "/models"))
        os.makedirs("models")

    if model_definition["language"] == "python":
        sys.path.append(model_dir)
        import model_modules

        if args.mode == "train":
            model_modules.training.train(data_conf, model_conf)
        elif args.mode == "evaluate":
            model_modules.scoring.evaluate(data_conf, model_conf)
        else:
            raise Exception("Unsupported mode used: " + args.mode)

    elif model_definition["language"] == "sql":
        if args.mode == "train":
            train_sql(model_dir, data_conf, model_conf)
        elif args.mode == "evaluate":
            evaluate_sql(model_dir, data_conf, model_conf)
        else:
            raise Exception("Unsupported mode used: " + args.mode)

    else:
        raise Exception("Unsupported cli language: {}".format(model_definition["language"]))


def evaluate_sql(model_dir, data_conf, model_conf):
    from teradataml import create_context
    from teradataml.dataframe.dataframe import DataFrame
    from teradataml.context.context import get_connection

    create_context(host=data_conf["hostname"],
                   username=os.environ["TD_USERNAME"],
                   password=os.environ["TD_PASSWORD"])

    print("Starting evaluation...")

    sql_file = model_dir + "/model_modules/scoring.sql"
    jinja_ctx = {"data_conf": data_conf, "model_conf": model_conf}

    execute_sql_script(get_connection(), sql_file, jinja_ctx)

    print("Finished evaluation")

    stats = DataFrame(data_conf["metrics_table"]).to_pandas()
    metrics = dict(zip(stats.key, stats.value))

    with open("models/evaluation.json", "w+") as f:
        json.dump(metrics, f)


def train_sql(model_dir, data_conf, model_conf):
    from teradataml import create_context
    from teradataml.context.context import get_connection

    create_context(host=data_conf["hostname"],
                   username=os.environ["TD_USERNAME"],
                   password=os.environ["TD_PASSWORD"])

    print("Starting training...")

    sql_file = model_dir + "/model_modules/training.sql"
    jinja_ctx = {"data_conf": data_conf, "model_conf": model_conf}

    execute_sql_script(get_connection(), sql_file, jinja_ctx)

    print("Finished training")

    print("Saved trained model")


def template_sql_script(filename, jinja_ctx):
    with open(filename) as f:
        template = Template(f.read())

    return template.render(jinja_ctx)


def execute_sql_script(conn, filename, jinja_ctx):
    script = template_sql_script(filename, jinja_ctx)

    stms = script.split(';')

    for stm in stms:
        stm = stm.strip()
        if stm:
            logging.info("Executing statement: {}".format(stm))

            try:
                conn.execute(stm)
            except Exception as e:
                if stm.startswith("DROP"):
                    logging.warning("Ignoring DROP statement exception")
                else:
                    raise e

if __name__ == "__main__":
    main()
