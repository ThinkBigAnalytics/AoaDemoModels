
DATABASE {{ data_conf.schema }};

DROP TABLE {{ data_conf.model_table }};

CREATE MULTISET TABLE {{ data_conf.model_table }}
AS (
  SELECT * FROM NaiveBayesTextClassifierTrainer (
    ON (
      SELECT * FROM NaiveBayesTextClassifierInternal (
        ON ( SELECT *
        FROM {{ data_conf.data_table }}
        ) AS "input" PARTITION BY category
        USING
        TokenColumn ('token')
       	ModelType ('{{ model_conf.hyperParameters.model_type }}')
        DocIDColumns ('customerid')
        DocCategoryColumn ('category')
      ) AS alias_1
    ) PARTITION BY 1
  ) AS A 
)
WITH DATA;