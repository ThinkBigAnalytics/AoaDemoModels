import json


def save_metadata(metadata_df, save_evaluation_metrics=False):
    """
    create statistic summaries based on the provided dataframe from training or evaluation

    partitions.json is {
        "<partition1 key>": <partition1_metadata>,
        "<partition2 key>": <partition2_metadata>,
        ...
    }

    data_stats.json is {
        "num_rows": <num_rows>,
        "num_partitions": <num_partitions>
    }

    :param metadata_df: pandas dataframe containing ["partition_id", "partition_metadata", "num_rows"]
    :param save_evaluation_metrics: if this is evaluation, save evaluation metrics.json also (partition_metadata must contain it)
    :return: None
    """
    metadata_dict = {r["partition_id"]: json.loads(r["partition_metadata"])
                     for r in metadata_df.to_dict(orient='records')}

    with open("artifacts/output/partitions.json", 'w+') as f:
        json.dump(metadata_dict, f, indent=2)

    num_rows = int(metadata_df["num_rows"].sum())
    data_metadata = {
        "num_rows": num_rows,
        "num_partitions": int(metadata_df.shape[0])
    }

    with open("artifacts/output/data_stats.json", 'w+') as f:
        json.dump(data_metadata, f, indent=2)

    if save_evaluation_metrics:
        # in future version we will add support for normalizing all metrics from all partitions
        metrics = {"n/a": 0}
        with open("artifacts/output/metrics.json", 'w+') as f:
            json.dump(metrics, f, indent=2)


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
