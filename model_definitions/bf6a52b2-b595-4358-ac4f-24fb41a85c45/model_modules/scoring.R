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

Connect2Vantage <- function() {
    # Create Teradata Vantage connection using tdplyr
    con <- td_create_context(host = Sys.getenv("AOA_CONN_HOST"),
                             uid = Sys.getenv("AOA_CONN_USERNAME"),
                             pwd = Sys.getenv("AOA_CONN_PASSWORD"),
                             dType = 'native'
    )

    # Set connection context
    td_set_context(con)

    con
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
    con <- Connect2Vantage()

    # Create tibble from table in Teradata Vantage
    table <- tbl(con, data_conf$table)

    # Create dataframe from tibble, selecting the necessary columns and mutating integer64 to integers
    data <- table %>% select(c("NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age")) %>% mutate(across(where(bit64::is.integer64), as.integer)) %>% as.data.frame()

    # The model object will be obtain from the environment as it has already been initialised using 'initialise_model'
    probs <- predict(model, data, na.action = na.pass, type = "response")
    score <- as.integer(ifelse(probs > 0.5, 1, 0))
    print("Finished batch scoring model...")

    # create result dataframe and store in Teradata Vantage
    score_df <- as.data.frame(unlist(score))
    colnames(score_df) <- c("Prediction")
    patientIds <- table %>% select("PatientId") %>% mutate(across(where(bit64::is.integer64), as.integer)) %>% as.data.frame()
    score_df$PatiendId <- patientIds$PatientId
    copy_to(con, score_df, name=data_conf$predictions, overwrite=TRUE)
    print("Saved batch predictions...")
}

initialise_model <- function() {
    print("Loading model...")
    model <- readRDS("artifacts/input/model.rds")
}

evaluate <- function(data_conf, model_conf, ...) {
    model <- initialise_model()
    print("Evaluating model...")

    suppressPackageStartupMessages(LoadPackages())

    # Connect to Vantage
    con <- Connect2Vantage()

    # Create tibble from table in Vantage
    table <- tbl(con, data_conf$table)

    # Create dataframe from tibble, selecting the necessary columns and mutating integer64 to integers
    data <- table %>% select(c("NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes")) %>% mutate(across(where(bit64::is.integer64), as.integer)) %>% as.data.frame()

    probs <- predict(model, data, na.action = na.pass, type = "response")
    preds <- as.integer(ifelse(probs > 0.5, 1, 0))

    cm <- confusionMatrix(table(preds, data$HasDiabetes))

    png("artifacts/output/confusion_matrix.png", width = 860, height = 860)
    fourfoldplot(cm$table)
    dev.off()

    preds$pred <- preds
    metrics <- cm$overall

    write(jsonlite::toJSON(metrics, auto_unbox = TRUE, null = "null", keep_vec_names=TRUE), "artifacts/output/metrics.json")
}
