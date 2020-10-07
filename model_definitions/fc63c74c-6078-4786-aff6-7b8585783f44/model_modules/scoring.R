library(methods)
library(gbm)
library(jsonlite)
library(caret)

score <- function(data_conf, model_conf, ...) {
    print("Batch scoring model...")
    data <- read.csv(url(data_conf[['url']]))
    new_data <- data[-c(9)]
    colnames(new_data) <- c("NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age")
    score <- predict(model, new_data, na.action = na.pass, type = "response")
    # The score is printed out but for instance it could be saved into a table, file, etc.
    print(score)
}

initialise_model <- function() {
    print("Loading model...")
    model <- readRDS("artifacts/input/model.rds")
}

evaluate <- function(data_conf, model_conf, ...) {
    print(
    model <- initialise_model()

    data <- read.csv(url(data_conf[['url']]))
    colnames(data) <- c("NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes")

    probs <- predict(model, data, na.action = na.pass, type = "response")
    preds <- as.integer(ifelse(probs > 0.5, 1, 0))

    cm <- confusionMatrix(table(preds, data$HasDiabetes))

    png("artifacts/output/confusion_matrix.png", width = 860, height = 860)
    fourfoldplot(cm$table)
    dev.off()

    preds$pred = preds
    metrics <- cm$overall

    write(jsonlite::toJSON(metrics, auto_unbox = TRUE, null = "null", keep_vec_names=TRUE), "artifacts/output/metrics.json")
}
