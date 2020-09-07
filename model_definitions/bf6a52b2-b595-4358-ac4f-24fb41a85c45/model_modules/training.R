library("gbm")


train <- function(data_conf, model_conf, ...) {

    data <- read.csv(url(data_conf[['url']]))
    colnames(data) <- c("NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes")

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
