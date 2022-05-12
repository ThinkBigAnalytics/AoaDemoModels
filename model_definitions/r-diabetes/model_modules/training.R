LoadPackages <- function() {
    library("gbm")
    library("DBI")
    library("dplyr")
    library("tdplyr")

}

suppressPackageStartupMessages(LoadPackages())

train <- function(data_conf, model_conf, ...) {
    # Connect to Vantage
    con <- aoa_create_context()

    table <- tbl(con, sql(data_conf$sql))

    # Create dataframe from tibble, selecting the necessary columns and mutating integer64 to integers
    # select both the feature and target columns (ignorning e.g. entity key)
    columns <- unlist(c(data_conf$featureNames, data_conf$targetNames), use.name = TRUE)
    data <- table %>% select(all_of(columns)) %>% mutate(
                       NumTimesPrg = as.integer(NumTimesPrg),
                       PlGlcConc = as.integer(PlGlcConc),
                       BloodP = as.integer(BloodP),
                       SkinThick = as.integer(SkinThick),
                       TwoHourSerIns = as.integer(TwoHourSerIns),
                       HasDiabetes = as.integer(HasDiabetes)) %>% as.data.frame()

    # Load hyperparameters from model configuration
    hyperparams <- model_conf[["hyperParameters"]]

    print("Training model...")

    # Train model
    model <- gbm(HasDiabetes~.,
                 data=data,
                 shrinkage=hyperparams$shrinkage,
                 distribution = 'bernoulli',
                 cv.folds=hyperparams$cv.folds,
                 n.trees=hyperparams$n.trees,
                 verbose=FALSE)

    print("Model Trained!")

    # Get optimal number of iterations
    best.iter <- gbm.perf(model, plot.it=FALSE, method="cv")

    # clean the model (R stores the dataset on the model..
    model$data <- NULL

    # how to save only best.iter tree?
    # model$best.iter <- best.iter
    # model$trees <- light$trees[best.iter]

    # Save trained model
    print("Saving trained model...")
    saveRDS(model, "artifacts/output/model.rds")
}
