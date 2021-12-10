from teradataml import DataFrame


def get_joined_models_df(data_table: str,
                         model_artefacts_table: str,
                         model_version: str,
                         partition_id: str = "partition_id"):
    """
    Joins the dataset which is to be used for scoring/evaluation with the model artefacts and appends the model_artefact
    to the first row with the column name 'model_artefact'.

    Args:
        data_table: the table/view of the dataset to join
        model_artefacts_table: the model artefacts table where the model artefacts are stored
        model_version: the model version to use from the model artefacts
        partition_id: the dataset partition_id

    Returns:
        DataFrame
    """
    query = f"""
    SELECT d.*, CASE WHEN n_row=1 THEN m.model_artefact ELSE null END AS model_artefact 
        FROM (SELECT x.*, ROW_NUMBER() OVER (PARTITION BY x.{partition_id} ORDER BY x.{partition_id}) AS n_row FROM {data_table} x) AS d
        LEFT JOIN {model_artefacts_table} m
        ON d.{partition_id} = m.partition_id
        WHERE m.model_version = '{model_version}'
    """

    return DataFrame(query=query)
