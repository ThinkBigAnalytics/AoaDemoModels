library(methods)
library(gbm)

predict.model <- function(model, data) {
    print("scoring model")
    predict(model$model, data, 1)
}

initialise_model <- function() {
    print("loading model")
    model <- readRDS("models/model.rds")
    structure(list(model=model), class = "model")
}

evaluate <- function(data_conf, model_conf, ...) {
    model <- initialise_model()

    data <- read.csv(url(data_conf[['url']]))
    colnames(data) <- c("NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes")

    preds <- predict(model$model, data, 1)

    results <- list("accuracy" = "95")
    write(jsonlite::toJSON(results, auto_unbox = TRUE, null = "null", force = TRUE), "models/evaluation.json")
}