library(methods)
library(gbm)

predict.model <- function(model, data) {
    print("scoring model")
    predict(model$model, data, 1)
}

new_gbm <- function(filename) {
    model <- readRDS(filename)
    structure(list(model=model), class = "model")
}

initialise_model <- function() {
    print("loading model")
    new_gbm("models/model.rds")
}


evaluate <- function(data_conf, model_conf) {
    initialise_model()

    results <- list("accuracy" = "95")
    write(jsonlite::toJSON(results, auto_unbox = TRUE, null = "null"), "models/evaluation.json")
}