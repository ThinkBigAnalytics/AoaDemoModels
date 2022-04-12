LoadPackages <- function() {
    library("gbm")
    library("DBI")
    library("dplyr")
    library("tdplyr")

}

suppressPackageStartupMessages(LoadPackages())

Connect2Vantage <- function() {
    # Create Vantage connection using tdplyr
    con <- td_create_context(host = Sys.getenv("AOA_CONN_HOST"),
                             uid = Sys.getenv("AOA_CONN_USERNAME"),
                             pwd = Sys.getenv("AOA_CONN_PASSWORD"),
                             dType = 'native'
    )

    # Set connection context
    td_set_context(con)

    con
}

train <- function(data_conf, model_conf, ...) {
    print("Training model...")

    # Connect to Vantage
    con <- Connect2Vantage()

    # Create tibble from table in Vantage
    if ("schema" %in% data_conf) {
        table_name <- in_schema(data_conf$schema, data_conf$table)
    } else {
        table_name <- data_conf$table
    }
    table <- tbl(con, table_name)

    # Create dataframe from tibble, selecting the necessary columns and mutating integer64 to integers
    data <- table %>% select(c("NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes")) %>%
      mutate(NumTimesPrg = as.integer(NumTimesPrg), PlGlcConc = as.integer(PlGlcConc), BloodP = as.integer(BloodP), SkinThick = as.integer(SkinThick), TwoHourSerIns = as.integer(TwoHourSerIns), HasDiabetes = as.integer(HasDiabetes)) %>%
      as.data.frame()

    # Load hyperparameters from model configuration
    hyperparams <- model_conf[["hyperParameters"]]

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
