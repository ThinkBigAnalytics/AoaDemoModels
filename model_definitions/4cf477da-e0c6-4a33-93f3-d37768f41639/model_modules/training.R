LoadPackages <- function() {
    library("h2o")
    library("DBI")
    library("dplyr")
    library("tdplyr")
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

    features <- c('age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'y')

    # Create dataframe from tibble, selecting the necessary columns and mutating as necessary
    print("Loading dataset...")
    data <- table %>% select(all_of(features)) %>% as.data.frame()
    data$age <- as.integer(data$age)
    data$job <- as.factor(data$job)
    data$marital <- as.factor(data$marital)
    data$education <- as.factor(data$education)
    data$default <- as.factor(data$default)
    data$balance <- as.integer(data$balance)
    data$housing <- as.factor(data$housing)
    data$loan <- as.factor(data$loan)
    data$y <- as.factor(data$y)
    data

    # Load hyperparameters from model configuration
    hyperparams <- model_conf[["hyperParameters"]]

    # Initialize and convert dataframe to h2o
    print("Initializing h2o...")
    h2o.init(nthreads = -1)
    train.hex <- as.h2o(data)
    splits <- h2o.splitFrame(train.hex, 0.75, seed=as.integer(Sys.time()))
    train <- splits[[1]]
    target <- "y"
    predictors <- setdiff(names(train), target)

    # Train model
    print("Training started...")
    aml <- h2o.automl(
        x = predictors,
        y = target,
        training_frame = train,
        max_models = hyperparams$max_models,
        seed = hyperparams$seed
    )
    model <- aml@leader
    print("Model Trained!")

    # Save trained model
    print("Saving trained model...")

    # Save trained model in h2o format
    output.dir <- getwd()
    path.value <- file.path(output.dir, "artifacts/output")
    h2o.saveModel(object = model, path = path.value, force = TRUE)
    name <- file.path(path.value, "model.h2o") # destination file name at the same folder location
    file.rename(file.path(path.value, model@model_id), name)

    # Save trained model as h2o mojo
    mojo <- h2o.download_mojo(model, path=path.value, get_genmodel_jar=TRUE)
    name <- file.path(path.value, "mojo.zip") # destination file name at the same folder location
    file.rename(file.path(path.value, mojo), name)

    # Convert mojo to pmml -> AutoML cannot be converted to PMML
    #cmd <- sprintf("wget https://aoa-public-files.s3.amazonaws.com/jpmml-h2o-executable-1.1-SNAPSHOT.jar && java -jar ./jpmml-h2o-executable-1.1-SNAPSHOT.jar --mojo-input %s --pmml-output %s", name, file.path(path.value, "model.pmml"))
    #result <- try(system(cmd, intern = TRUE))
    #print(result)
    print("Trained model saved")
}
