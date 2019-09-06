#!/usr/bin/env python

import json
import logging
import uuid
import sys
import shutil
import os
import collections

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Model Repository Operations')
    parser.add_argument('-a', '--add', action='store_true', help='Add Model')

    args = parser.parse_args()

    if args.add:

        model = collections.OrderedDict()
        model["id"] = str(uuid.uuid4())
        model["name"] = input("Model Name: ")
        model["description"] = input("Model Description: ")
        model["language"] = input("Model Language: ")

        if model["language"] in ["python", "R", "sql"]:
            create_model_structure(model)
        else:
            logging.error("Only python, R and sql models currently supported.")
            exit(1)

    else:
        logging.error("Only --add option is currently supported")
        exit(1)


def create_model_structure(model):
    logging.info("Creating model structure for model: {}".format(model["id"]))

    base_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
    model_dir = base_path + "./model_definitions/" + model["id"]
    template_dir = base_path + "./model_templates/" + model["language"]

    shutil.copytree(template_dir, model_dir)

    with open(model_dir + "/model.json", 'w') as f:
        json.dump(model, f, indent=4)


if __name__== "__main__":
    main()
