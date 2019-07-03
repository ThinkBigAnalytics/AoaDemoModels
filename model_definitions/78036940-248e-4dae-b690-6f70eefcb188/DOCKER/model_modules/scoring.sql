DROP TABLE csi_telco_churn_predict_test;
DROP TABLE telco_count_output;
DROP TABLE telco_stat_output;
DROP TABLE telco_acc_output;


CREATE TABLE csi_telco_churn_predict_test
AS
(SELECT *
    FROM NaiveBayesTextClassifierPredict@coprocessor (
    ON (SELECT *
        FROM model_dataset_test
    ) AS predicts 
    PARTITION BY customerid
    ON csi_telco_churn_model AS model DIMENSION
    USING
    InputTokenColumn ('token')
    ModelType ('Bernoulli')
    DocIDColumns ('customerid')
    TopK (1)
  ) AS dt
)
WITH DATA;

REPLACE VIEW csi_telco_churn_results AS 
    SELECT P.customerid, P.prediction, P.loglik, T.category actual
        FROM csi_telco_churn_predict_test P 
        JOIN model_dataset_test T ON P.customerid = T.customerid;


SELECT * FROM ConfusionMatrix (
  ON csi_telco_churn_results
    PARTITION BY 1
    OUT TABLE CountTable (telco_count_output)
    OUT TABLE StatTable (telco_stat_output)
    OUT TABLE AccuracyTable(telco_acc_output)
    USING
    ObsColumn ('actual')
    PredictColumn ('prediction')
) AS dt;