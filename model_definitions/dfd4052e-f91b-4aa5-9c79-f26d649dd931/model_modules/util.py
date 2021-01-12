from teradataml.dataframe.dataframe import DataFrame

import json


def save_evaluation_metrics(partition_df, metrics):
    """
    :param partition_df: teradata dataframe containing at least ["partition_id", "partition_metadata", "num_rows"]
    :return:
    """
    total_rows = int(partition_df.select(["num_rows"]).sum().to_pandas().iloc[0])

    metrics_sql = [f"SUM(CAST(partition_metadata AS JSON).JSONExtractValue('$.metrics.{metric}') * num_rows/{total_rows}) AS {metric}" for metric in metrics]
    joined_metrics_sql = ','.join(metrics_sql)
    metrics = DataFrame.from_query(f"SELECT {joined_metrics_sql} FROM {partition_df._table_name}").to_pandas()

    metrics = { metric : "{:.2f}".format(metrics.iloc[0][metric]) for metric in metrics }

    with open("artifacts/output/metrics.json", 'w+') as f:
        json.dump(metrics, f, indent=2)


def save_metadata(partition_df):
    """
    create statistic summaries based on the provided dataframe produced via training or evaluation

    partitions.json is {
        "<partition1 key>": <partition1_metadata>,
        "<partition2 key>": <partition2_metadata>,
        ...
    }

    data_stats.json is {
        "num_rows": <num_rows>,
        "num_partitions": <num_partitions>
    }

    :param partition_df: teradata dataframe containing at least ["partition_id", "partition_metadata", "num_rows"]
    :return: None
    """

    total_rows = int(partition_df.select(["num_rows"]).sum().to_pandas().iloc[0])

    metadata_df = partition_df.select(["partition_id", "partition_metadata", "num_rows"]).to_pandas()

    metadata_dict = {r["partition_id"]: json.loads(r["partition_metadata"])
                     for r in metadata_df.to_dict(orient='records')}

    with open("artifacts/output/partitions.json", 'w+') as f:
        json.dump(metadata_dict, f, indent=2)

    data_metadata = {
        "num_rows": total_rows,
        "num_partitions": int(metadata_df.shape[0])
    }

    with open("artifacts/output/data_stats.json", 'w+') as f:
        json.dump(data_metadata, f, indent=2)


def cleanup_cli(model_version):
    """
    cli uses model version of "cli" always. We need to cleanup models table between runs.
    A better solution would be for the cli to write to a different table completely and just "recreate" on each run

    :param model_version: the model version being executed
    :return: None
    """
    if model_version == "cli":
        from teradataml.context.context import get_connection
        get_connection().execute("DELETE FROM aoa_sto_models WHERE model_version='cli'")
