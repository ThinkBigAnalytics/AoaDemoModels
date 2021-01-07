# STO Micro Modelling Example

This model is a purely ficticious regression example to show how to use STOs to train, evaluation and score micro models (individual models per data partition in Teradata).

The micro models are stored in Teradata in the following schema.


```sql
CREATE MULTISET TABLE aoa_sto_models, FALLBACK ,
     NO BEFORE JOURNAL,
     NO AFTER JOURNAL,
     CHECKSUM = DEFAULT,
     DEFAULT MERGEBLOCKRATIO,
     MAP = TD_MAP1
     (
        partition_id VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
        model_version VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
        num_rows BIGINT,
        partition_metadata JSON CHARACTER SET UNICODE,
        model_artefact CLOB
     )
UNIQUE PRIMARY INDEX (partition_id, model_version);
```

## Dataset Description
The dataset is automatically generated using a simple polynomial and some random columns also. This allows us to fit a regression model to it but not be too simple either. We only created 10 partitions in this example.

Training dataset configuration

    table: STO_SYNTHETIC_TRAIN_V
    
Evaluation dataset configuration

    table: STO_SYNTHETIC_TEST_V
