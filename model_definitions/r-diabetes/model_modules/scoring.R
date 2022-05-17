library(methods)
library(gbm)
library(jsonlite)
library(caret)

LoadBatchScoringPackages <- function() {
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

score.batch <- function(data_conf, model_conf, model_version, job_id, ...) {
    model <- initialise_model()
    print("Batch scoring model...")

    suppressPackageStartupMessages(LoadBatchScoringPackages())

    # Connect to Teradata Vantage
    con <- aoa_create_context()

    table <- tbl(con, sql(data_conf$sql))

    # Create dataframe from tibble, selecting the necessary columns and mutating integer64 to integers
    data <- table %>% mutate(PatientId = as.integer(PatientId),
                             NumTimesPrg = as.integer(NumTimesPrg),
                             PlGlcConc = as.integer(PlGlcConc),
                             BloodP = as.integer(BloodP),
                             SkinThick = as.integer(SkinThick),
                             TwoHourSerIns = as.integer(TwoHourSerIns)) %>% as.data.frame()

    # The model object will be obtain from the environment as it has already been initialised using 'initialise_model'
    probs <- predict(model, data, na.action = na.pass, type = "response")
    score <- as.integer(ifelse(probs > 0.5, 1, 0))
    print("Finished batch scoring model...")

    # create result dataframe and store in Teradata Vantage
    pred_df <- as.data.frame(unlist(score))
    colnames(pred_df) <- c("HasDiabetes")
    pred_df$PatientId <- data$PatientId
    pred_df$job_id <- job_id

    # tdplyr doesn't match column names on append.. and so to match / use same table schema as for byom predict
    # example (see README.md), we must add empty json_report column and change column order manually (v17.0.0.4)
    # CREATE MULTISET TABLE pima_patient_predictions
    # (
    #     job_id VARCHAR(255), -- comes from airflow on job execution
    #     PatientId BIGINT,    -- entity key as it is in the source data
    #     HasDiabetes BIGINT,   -- if model automatically extracts target
    #     json_report CLOB(1048544000) CHARACTER SET UNICODE  -- output of
    # )
    # PRIMARY INDEX ( job_id );
    pred_df$json_report <- ""
    pred_df <- pred_df[, c("job_id", "PatientId", "HasDiabetes", "json_report")]

    copy_to(con, pred_df,
            name=dbplyr::in_schema(data_conf$predictions$database, data_conf$predictions$table),
            types = c("varchar(255)", "bigint", "bigint", "clob"),
            append=TRUE)
    print("Saved batch predictions...")
}

initialise_model <- function() {
    print("Loading model...")
    model <- readRDS("artifacts/input/model.rds")
}
