LoadPackages <- function() {
    library("h2o")
    library("DBI")
    library("dplyr")
    library("tdplyr")
    library("caret")
}

suppressPackageStartupMessages(LoadPackages())

Connect2Vantage <- function() {
    # Create Vantage connection using tdplyr
    con <- td_create_context(
        host = Sys.getenv("AOA_CONN_HOST"),
        uid = Sys.getenv("AOA_CONN_USERNAME"),
        pwd = Sys.getenv("AOA_CONN_PASSWORD"),
        dType = 'native'
    )

    # Set connection context
    td_set_context(con)

    con
}

initialise_model <- function() {
    print("Loading h2o model...")
    h2o.init(nthreads = -1)
    output.dir <- getwd()
    path.value <- file.path(output.dir, "artifacts/input")
    name <- file.path(path.value, "model.h2o")
    model <- h2o.loadModel(name)
}

score.restful <- function(model, data, ...) {
    print("Scoring model...")
    
    data$age <- as.integer(data$age)
    data$job <- as.factor(data$job)
    data$marital <- as.factor(data$marital)
    data$education <- as.factor(data$education)
    data$default <- as.factor(data$default)
    data$balance <- as.integer(data$balance)
    data$housing <- as.factor(data$housing)
    data$loan <- as.factor(data$loan)
    
    data_df <- as.h2o(data)
    score <- h2o.predict(model, data_df)
    score_df <- as.data.frame(score)
    score_df
}

score.batch <- function(data_conf, model_conf, model_version, ...) {
    model <- initialise_model()
    print("Batch scoring model...")

    suppressPackageStartupMessages(LoadPackages())

    # Connect to Teradata Vantage
    con <- Connect2Vantage()

    # Create tibble from table in Teradata Vantage
    if ("schema" %in% data_conf) {
        table_name <- in_schema(data_conf$schema, data_conf$table)
    } else {
        table_name <- data_conf$table
    }
    table <- tbl(con, table_name)

    predictors <- c('age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'y')
    target <- 'y'

    # Create dataframe from tibble, selecting the necessary columns and mutating as necessary
    print("Loading dataset...")
    data <- table %>% select(all_of(predictors)) %>% as.data.frame()
    data$age <- as.integer(data$age)
    data$job <- as.factor(data$job)
    data$marital <- as.factor(data$marital)
    data$education <- as.factor(data$education)
    data$default <- as.factor(data$default)
    data$balance <- as.integer(data$balance)
    data$housing <- as.factor(data$housing)
    data$loan <- as.factor(data$loan)
    data$y <- as.factor(data$y)
    
    # Convert dataframe to h2o
    data <- as.h2o(data)

    # The model object will be obtain from the environment as it has already been initialised using 'initialise_model'
    score <- h2o.predict(model, data)
    print("Finished batch scoring model")

    # create result dataframe and store in Teradata Vantage
    score_df <- as.data.frame(score$predict)
    if ("schema" %in% data_conf) {
        predictions_table_name <- SQL(sprintf("%s.%s", data_conf$schema, data_conf$predictions))
    } else {
        predictions_table_name <- data_conf$predictions
    }
    copy_to(con, score_df, name=predictions_table_name, overwrite=TRUE)
    print("Saved batch predictions")
}
