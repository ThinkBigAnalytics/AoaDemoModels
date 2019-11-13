
DROP TABLE {{ data_conf.predictions_table }};
DROP TABLE {{ data_conf.metrics_table }};
DROP TABLE {{ data_conf.metrics_table }}_COUNTS;
DROP TABLE {{ data_conf.metrics_table }}_STATS;
DROP TABLE {{ data_conf.metrics_table }}_ACC;

CREATE TABLE {{ data_conf.predictions_table }}
AS (
    SELECT * FROM XGBoostPredict(
    	ON "{{ data_conf.data_table }}" AS "input"
    	PARTITION BY ANY
    	ON "{{ model_table }}" AS ModelTable
    	DIMENSION
    	ORDER BY "tree_id","iter","class_num"
    	USING
    	IdColumn('idx')
    	Accumulate('hasdiabetes')
    ) as sqlmr
)
WITH DATA;


SELECT * FROM ConfusionMatrix(
    ON "{{ data_conf.predictions_table }}" AS "input"
    PARTITION BY 1
    OUT TABLE CountTable("{{ data_conf.metrics_table }}_COUNTS")
    OUT TABLE StatTable("{{ data_conf.metrics_table }}_STATS")
    OUT TABLE AccuracyTable("{{ data_conf.metrics_table }}_ACC")
    USING
    ObservationColumn('hasdiabetes')
    PredictColumn('prediction')
) as sqlmr;

-- confusion matrix outputs 3 tables with the _1,_2,_3 appended. We only want to keep the accuracy metrics this time.

DROP TABLE {{ data_conf.metrics_table }}_COUNTS;
DROP TABLE {{ data_conf.metrics_table }}_ACC;

RENAME TABLE {{ data_conf.metrics_table }}_STATS TO {{ data_conf.metrics_table }};