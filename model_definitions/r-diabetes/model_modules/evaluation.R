
LoadPackages <- function() {
  library("methods")
  library("jsonlite")
  library("caret")
  library("gbm")
  library("DBI")
  library("dplyr")
  library("tdplyr")
}

evaluate <- function(data_conf, model_conf, ...) {
  model <- readRDS("artifacts/input/model.rds")
  print("Evaluating model...")

  suppressPackageStartupMessages(LoadPackages())

  # Connect to Vantage
  con <- aoa_create_context()

  table <- tbl(con, sql(data_conf$sql))

    # Create dataframe from tibble, selecting the necessary columns and mutating integer64 to integers
  data <- table %>% mutate(NumTimesPrg = as.integer(NumTimesPrg),
                                PlGlcConc = as.integer(PlGlcConc),
                                BloodP = as.integer(BloodP),
                                SkinThick = as.integer(SkinThick),
                                TwoHourSerIns = as.integer(TwoHourSerIns),
                                HasDiabetes = as.integer(HasDiabetes)) %>% as.data.frame()

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
