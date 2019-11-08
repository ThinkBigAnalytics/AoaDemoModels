
DROP TABLE {{ data_conf.model_table }};

SELECT * FROM XGBoost(
    ON "{{ data_conf.data_table }}" AS InputTable
    OUT TABLE OutputTable("{{ model_table }}")
    USING
    IdColumn('idx')
    MaxDepth('10')
    ResponseColumn('hasdiabetes')
    NumericInputs('numtimesprg','plglcconc','bloodp','skinthick','twohourserins','bmi','dipedfunc','age')
) as sqlmr
