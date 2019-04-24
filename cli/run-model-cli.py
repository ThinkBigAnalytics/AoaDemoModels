#!/usr/bin/env python

import json
import logging
import sys
import os

from importlib.machinery import SourceFileLoader

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Run model training or scoring locally')
    parser.add_argument('model_id', type=str, help='Which model_id to use')
    parser.add_argument('mode', type=str, help='The model (train or eval)')
    parser.add_argument('--data', type=str, help='Yaml file containing data configuration')

    args = parser.parse_args()

    base_path = os.path.dirname(os.path.realpath(__file__)) + "/../"

    model_dir = base_path + "./model_definitions/" + args.model_id + "/DOCKER"

    with open(model_dir + "/config.json", 'r') as f:
        model_conf = json.load(f)

    if args.data:
        with open(args.data, 'r') as f:
            data_conf = json.load(f)

    else:
        data_conf = {}

    module = SourceFileLoader("model", model_dir + "/model_modules/__init__.py").load_module()
    if args.mode == "train":
        module.training.train(data_conf, model_conf)

    elif args.mode == "eval":
        module.scoring.evaluate(data_conf, model_conf)

    else:
        raise Exception("Unsupported mode used: " + args.mode)


if __name__== "__main__":
    main()
