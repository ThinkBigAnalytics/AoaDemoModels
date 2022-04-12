
LoadPackages <- function() {
  library("methods")
  library("jsonlite")
  library("caret")
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

evaluate <- function(data_conf, model_conf, ...) {
  model <- readRDS("artifacts/input/model.rds")
  print("Evaluating model...")

  suppressPackageStartupMessages(LoadPackages())

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
