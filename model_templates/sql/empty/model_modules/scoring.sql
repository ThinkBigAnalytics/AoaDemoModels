
DATABASE {{ data_conf.schema }};

DROP TABLE {{ data_conf.predictions_table }};
DROP TABLE {{ data_conf.results_table }};

CREATE TABLE {{ data_conf.predictions_table }}
AS (
    SELECT * FROM XGBoostPredict(
    	ON "{{ data_conf.data_table }}" AS "input"
    	PARTITION BY ANY
    	ON "{{ data_conf.model_table }}" AS ModelTable
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
	OUT TABLE AccuracyTable("{{ data_conf.results_table }}")
	USING
	ObservationColumn('hasdiabetes')
	PredictColumn('prediction')
) as sqlmr