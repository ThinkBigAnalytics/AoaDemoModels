#!/usr/bin/env python3

import json
import logging
import uuid
import sys
import shutil
import os
import collections

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

base_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
template_catalog = base_path + "./model_templates/"
model_catalog = base_path + "./model_definitions/"


def main():

    if os.path.isdir(template_catalog) == False:
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
        model["supportedFrameworks"] = ["DOCKER"]
        print("These languages are supported: ",", ".join(str(x) for x in catalog.keys()))
        model_lang = input("Model Language: ")
        if model_lang not in catalog.keys():
            logging.error("Only %s models currently supported." % ", ".join(str(x) for x in catalog.keys()))
            exit(1)
        print("These templates are available for %s: " % model_lang,", ".join(str(x) for x in catalog[model_lang]))
        model_template = input("Template type (leave blank for the empty one): ")
        if not model_template:
            model_template = "empty"
        if model_template not in catalog[model_lang]:
            logging.error("Only %s templates currently supported." % ", ".join(str(x) for x in catalog[model_lang]))
            exit(1)
        
        model["language"] = model_lang
        model["template"] = model_template
        create_model_structure(model)
                
    else:
        logging.error("Only --add option is currently supported")
        exit(1)

def create_model_structure(model):
    logging.info("Creating model structure for model: {}".format(model["id"]))

    model_dir = model_catalog + model["id"]
    template_dir = template_catalog + model["language"] + "/" + model["template"]

    shutil.copytree(template_dir, model_dir)

    with open(model_dir + "/model.json", 'w') as f:
        json.dump(model, f, indent=4)

def get_template_catalog():
    catalog = {}
    for language in os.listdir(template_catalog):
        language_dir = template_catalog + "/" + language
        catalog[language] = []
        for template_type in os.listdir(language_dir):
            catalog[language].append(template_type)
    return catalog

if __name__== "__main__":
    main()
