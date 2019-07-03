DROP TABLE csi_telco_churn_model;

CREATE MULTISET TABLE csi_telco_churn_model 
AS (
  SELECT * FROM NaiveBayesTextClassifierTrainer (
    ON (
      SELECT * FROM NaiveBayesTextClassifierInternal (
        ON ( SELECT *
        FROM model_dataset_train
        ) AS "input" PARTITION BY category
        USING
        TokenColumn ('token')
       	ModelType ('Bernoulli')
        DocIDColumns ('customerid')
        DocCategoryColumn ('category')
      ) AS alias_1
    ) PARTITION BY 1
  ) AS alias_2 
)
WITH DATA;