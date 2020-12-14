# Sample Model 

Add you model details here..

## Dataset Description

## Training


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

## Evaluation

## Scoring