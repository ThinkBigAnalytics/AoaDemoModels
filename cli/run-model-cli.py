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
    parser.add_argument('-id', '--model_id', type=str, help='Which model_id to use (prompted to select if not provided)')
    parser.add_argument('-m', '--mode', type=str, help='The model (train or evaluate) (prompted to select if not provided)')
    parser.add_argument('-d', '--data', type=str, help='Json file containing data configuration (prompted to select if not provided)')

    args = parser.parse_args()

    base_path = os.path.dirname(os.path.realpath(__file__)) + "/../"

    if not args.model_id:
        model_id = get_model_id(base_path + "./model_definitions/")
    else:
        model_id = args.model_id

    if not args.mode:
        mode = input("Select mode (train, evaluate): ")
    else:
        mode = args.mode

    model_dir = base_path + "./model_definitions/" + model_id

    if not args.data:
        data = get_dataset_metadata(model_dir)
    else:
        data = args.data

    with open(data, 'r') as f:
        data_conf = json.load(f)

    with open(model_dir + "/model.json", 'r') as f:
        model_definition = json.load(f)

    with open(model_dir + "/config.json", 'r') as f:
        model_conf = json.load(f)

    if os.path.exists("models"):
        logging.info("Cleaning directory {} to store test model artefacts".format(os.getcwd() + "/models"))
        shutil.rmtree("models")

    else:
        logging.info("Creating directory {} to store test model artefacts".format(os.getcwd() + "/models"))
        os.makedirs("models")

    if model_definition["language"] == "python":
        sys.path.append(model_dir)
        import model_modules

        if mode == "train":
            model_modules.training.train(data_conf, model_conf)
        elif mode == "evaluate":
            model_modules.scoring.evaluate(data_conf, model_conf)
        else:
            raise Exception("Unsupported mode used: " + mode)

    elif model_definition["language"] == "sql":
        if mode == "train":
            train_sql(model_dir, data_conf, model_conf)
        elif mode == "evaluate":
            evaluate_sql(model_dir, data_conf, model_conf)
        else:
            raise Exception("Unsupported mode used: " + mode)

    else:
        raise Exception("Unsupported cli language: {}".format(model_definition["language"]))


def get_model_id(models_dir):
    catalog = {}
    index = 0

    for model in os.listdir(models_dir):
        if os.path.exists(models_dir + model + "/model.json"):
            with open(models_dir + model + "/model.json", 'r') as f:
                model_definition = json.load(f)

                catalog[index] = model_definition

                index += 1

    for key in catalog:
        print("({}) {}".format(key, catalog[key]["name"]))

    index = input("Select Model: ")

    return catalog[int(index)]["id"]


def get_dataset_metadata(model_dir):
    catalog = {}
    index = 0

    if not os.path.exists(model_dir + "/.cli/datasets"):
        raise Exception("No .cli/datasets exist in model folder. You must specify dataset metadata via -d argument.")

    base_path = model_dir + "/.cli/datasets/"
    for filename in os.listdir(base_path):
        with open(base_path + filename, 'r') as f:
            catalog[index] = filename
            index += 1

    for key in catalog:
        print("({}) {}".format(key, catalog[key]))

    index = input("Select Dataset Metadata: ")

    return base_path + catalog[int(index)]


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
