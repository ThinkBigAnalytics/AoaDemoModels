library(methods)

predict.model <- function(model, data) {
    print("scoring model")
}

initialise_model <- function() {
    print("loading model")

    #return model
    data.frame()
}

evaluate <- function(data_conf, model_conf, ...) {
    initialise_model()

    results <- list("accuracy" = "95")
    write(jsonlite::toJSON(results, auto_unbox = TRUE, null = "null"), "models/evaluation.json")
}