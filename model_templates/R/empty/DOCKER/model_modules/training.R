

train <- function(data_conf, model_conf, ...) {
    hyperparams <- model_conf[["hyperParameters"]]

    # data <- load_data()

    # create model
    model = data.frame()

    saveRDS(model, "models/model.rds")
}
