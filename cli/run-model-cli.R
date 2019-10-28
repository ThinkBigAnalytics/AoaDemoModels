#!/usr/bin/env Rscript

# https://github.com/ThinkBigAnalytics/AoaCoreService/issues/78
# Code to be moved into centralised aoa cli

library("jsonlite")
library("argparse")


parser <- ArgumentParser()

parser$add_argument("model_id",  help="The Model Id to run")
parser$add_argument("mode", help="The mode (train or evaluate)")
parser$add_argument('data', help="Json file containing data configuration")
args <- parser$parse_args()

model_id <- args$model_id
mode <- tolower(args$mode)

# direction of model, obtained form args[1]
model_dir <- paste("./model_definitions/",model_id, sep = "")

# get the configuration of the model
model_conf_dir <- paste(model_dir, "/config.json", sep = "")
model_conf <- jsonlite::read_json(model_conf_dir)

# get the json file to get the data
data_conf <- read_json(args$data)



# define the path of model modules
scripts_path <- paste(model_dir, "/model_modules/", sep = "")
training_path <- paste(scripts_path,"training.R", sep = "")
scoring_path <- paste(scripts_path,"scoring.R", sep = "")


if (mode == "train") {
    source(training_path)
    train(data_conf, model_conf)
} else if (mode == "evaluate") {
    source(scoring_path)
    evaluate(data_conf, model_conf)
} else {
    message("The mode was invalid")

    quit(status=1)
}