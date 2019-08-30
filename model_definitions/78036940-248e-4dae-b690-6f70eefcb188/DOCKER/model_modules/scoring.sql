
DATABASE {{ data_conf.schema }};

DROP TABLE {{ data_conf.schema }}.csi_telco_churn_predict_test;
DROP TABLE {{ data_conf.schema }}.telco_count_output;
DROP TABLE {{ data_conf.schema }}.telco_stat_output;
DROP TABLE {{ data_conf.schema }}.telco_acc_output;
DROP VIEW {{ data_conf.schema }}.csi_telco_churn_results;

CREATE TABLE {{ data_conf.schema }}.csi_telco_churn_predict_test
AS
(SELECT *
    FROM NaiveBayesTextClassifierPredict@coprocessor (
    ON (SELECT *
        FROM {{ data_conf.data_table }}
    ) AS predicts 
    PARTITION BY customerid
    ON {{ data_conf.model_table }} AS model DIMENSION
    USING
    InputTokenColumn ('token')
    ModelType ('{{ model_conf.hyperParameters.model_type }}')
    DocIDColumns ('customerid')
    TopK (1)
  ) AS dt
)
WITH DATA;

CREATE VIEW {{ data_conf.schema }}.csi_telco_churn_results AS 
    SELECT P.customerid, P.prediction, P.loglik, T.category actual
        FROM {{ data_conf.schema }}.csi_telco_churn_predict_test P 
        JOIN (SELECT customerid, category FROM {{ data_conf.data_table }} GROUP BY customerid, category) T ON P.customerid = T.customerid;


SELECT * FROM ConfusionMatrix (
  ON {{ data_conf.schema }}.csi_telco_churn_results
    PARTITION BY 1
    OUT TABLE CountTable ({{ data_conf.schema }}.telco_count_output)
    OUT TABLE StatTable ({{ data_conf.schema }}.telco_stat_output)
    OUT TABLE AccuracyTable({{ data_conf.schema }}.telco_acc_output)
    USING
    ObsColumn ('actual')
    PredictColumn ('prediction')
) AS dt;