library(methods)
library(gbm)
library(jsonlite)
library(caret)

LoadPackages <- function() {
    library("gbm")
    library("DBI")
    library("dplyr")
    library("tdplyr")
}

score.restful <- function(model, data, ...) {
    print("Scoring model...")
    probs <- predict(model, data, na.action = na.pass, type = "response")
    score <- ifelse(probs > 0.5, 1, 0)
    score
}

score.batch <- function(data_conf, model_conf, model_version, ...) {
    model <- initialise_model()
    print("Batch scoring model...")

    suppressPackageStartupMessages(LoadPackages())

    # Connect to Teradata Vantage
    con <- aoa_create_context()

    table <- tbl(con, sql(data_conf$sql))

    # Create dataframe from tibble, selecting the necessary columns and mutating integer64 to integers
    data <- table %>% mutate(NumTimesPrg = as.integer(NumTimesPrg),
                             PlGlcConc = as.integer(PlGlcConc),
                             BloodP = as.integer(BloodP),
                             SkinThick = as.integer(SkinThick),
                             TwoHourSerIns = as.integer(TwoHourSerIns)) %>% as.data.frame()

    # The model object will be obtain from the environment as it has already been initialised using 'initialise_model'
    probs <- predict(model, data, na.action = na.pass, type = "response")
    score <- as.integer(ifelse(probs > 0.5, 1, 0))
    print("Finished batch scoring model...")

    # create result dataframe and store in Teradata Vantage
    score_df <- as.data.frame(unlist(score))
    colnames(score_df) <- c("Prediction")
    patientIds <- table %>% select("PatientId") %>% mutate(PatientId = as.integer(PatientId)) %>% as.data.frame()
    score_df$PatiendId <- patientIds$PatientId

    # dbplyr::in_schema(data_conf$predictions$database, data_conf$predictions$table)
    copy_to(con, score_df,
            name=dbplyr::in_schema(data_conf$predictions$database, "pima_predictions_r"),
            overwrite=TRUE)
    print("Saved batch predictions...")
}

initialise_model <- function() {
    print("Loading model...")
    model <- readRDS("artifacts/input/model.rds")
}
