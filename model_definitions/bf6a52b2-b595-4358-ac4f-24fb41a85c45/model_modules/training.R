library("gbm")
library("tdplyr")
library("dplyr")
library("dbplyr")

train <- function(data_conf, model_conf, ...) {
    print("Training model...")

    con <- td_create_context(host = data_conf[["host"]], uid=Sys.getenv("TD_USERNAME"), pwd=Sys.getenv("TD_PASSWORD"), dType = "native")

    data <- tbl(con, data_conf[["table"]])
    data <- as.data.frame(data)
    data <- data[c("NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes")]

    hyperparams <- model_conf[["hyperParameters"]]

    model <- gbm(HasDiabetes~.,
                 data=data,
                 shrinkage=hyperparams$shrinkage,
                 distribution = 'bernoulli',
                 cv.folds=hyperparams$cv.folds,
                 n.trees=hyperparams$n.trees,
                 verbose=FALSE)

    best.iter <- gbm.perf(model, plot.it=FALSE, method="cv")

    # clean the model (R stores the dataset on the model..
    model$data <- NULL

    # how to save only best.iter tree?
    # model$best.iter <- best.iter
    # model$trees <- light$trees[best.iter]

    saveRDS(model, "artifacts/output/model.rds")
}
