library("gbm")


train <- function(data_conf, model_conf, ...) {

    data <- read.csv(url(data_conf[['url']]))
    colnames(data) <- c("NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes")

    hyperparams <- model_conf[["hyperParameters"]]

    model <- gbm(HasDiabetes~NumTimesPrg+PlGlcConc+BloodP+SkinThick+TwoHourSerIns+BMI+DiPedFunc+Age,
            data=data,
            var.monotone=c(0,0,0,0,0,0,0,0),
            distribution="gaussian",
            n.trees=hyperparams$n.trees,
            shrinkage=hyperparams$shrinkage,
            interaction.depth=hyperparams$interaction.depth,
            bag.fraction=hyperparams$bag.fraction,
            train.fraction=hyperparams$train.fraction,
            n.minobsinnode=hyperparams$n.minobsinnode,
            keep.data=TRUE,
            cv.folds=hyperparams$cv.folds,
            verbose = FALSE)

    best.iter <- gbm.perf(model, plot.it=FALSE, method="cv")

    # clean the model
    light <- model
    light$trees <- light$trees[best.iter]
    light$data <- list()

    saveRDS(light, "models/model.rds")
}
