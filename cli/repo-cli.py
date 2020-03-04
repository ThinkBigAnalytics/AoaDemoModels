#!/usr/bin/env python3

# https://github.com/ThinkBigAnalytics/AoaCoreService/issues/78
# Code to be moved into centralised aoa cli

import json
import logging
import uuid
import sys
import shutil
import os
import collections

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
template_catalog = os.path.join(base_path, "model_templates/")
model_catalog = os.path.join(base_path, "model_definitions/")


def main():

    if not os.path.isdir(template_catalog):
        logging.error("Template directory is missing")
        exit(1)
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Model Repository Operations')
    parser.add_argument('-a', '--add', action='store_true', help='Add Model')

    args = parser.parse_args()

    if args.add:

        model = collections.OrderedDict()
        catalog = get_template_catalog()

        model["id"] = str(uuid.uuid4())
        model["name"] = input("Model Name: ")
        model["description"] = input("Model Description: ")

        print("These languages are supported: {0}".format(
            ", ".join(str(x) for x in catalog.keys())))
        model_lang = input("Model Language: ")
        if model_lang not in catalog.keys():
            logging.error("Only {0} model languages currently supported.".format(
                ", ".join(str(x) for x in catalog.keys())))
            exit(1)

        print("These templates are available for {0}: {1}".format(
            model_lang, ", ".join(str(x) for x in catalog[model_lang])))
        model_template = input(
            "Template type (leave blank for the default one): ")
        if not model_template:
            model_template = "empty"
        if model_template not in catalog[model_lang]:
            logging.error("Only {0} templates currently supported.".format(
                ", ".join(str(x) for x in catalog[model_lang])))
            exit(1)

        model["language"] = model_lang

        add_framework_specific_attributes(model, model_template)

        create_model_structure(model, model_template)

    else:
        logging.error("Only --add option is currently supported")
        exit(1)


def add_framework_specific_attributes(model, model_template):
    if model_template == "pyspark":
        model["automation"] = {
            "trainingEngine": "pyspark"
        }


def create_model_structure(model, model_template):
    logging.info("Creating model structure for model: {0}".format(model["id"]))

    model_dir = model_catalog + model["id"]
    template_dir = os.path.join(
        template_catalog, model["language"], model_template)

    shutil.copytree(template_dir, model_dir)

    with open(os.path.join(model_dir, "model.json"), 'w') as f:
        json.dump(model, f, indent=4)


def get_template_catalog():
    catalog = {}
    # get a list of subfolders in template_catalog and remove hidden subfolders
    subfolders = [f for f in os.listdir(template_catalog)
                  if os.path.isdir(os.path.join(template_catalog, f)) & (f[0] != '.')]
    for language in subfolders:
        language_dir = os.path.join(template_catalog, language)
        catalog[language] = []
        for template_type in os.listdir(language_dir):
            # skip hidden folders
            if template_type[0] != '.':
                catalog[language].append(template_type)
    return catalog


if __name__ == "__main__":
    main()
