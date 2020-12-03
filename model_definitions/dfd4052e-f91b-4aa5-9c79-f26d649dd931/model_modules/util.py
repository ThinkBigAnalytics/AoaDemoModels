import json


def save_metadata(df):
    # convert stats to dict and save to partitions.json
    metadata_df = df.select(["partition_id", "partition_metadata", "num_rows"]).to_pandas()
    metadata_dict = {r["partition_id"]: json.loads(r["partition_metadata"]) for r in
                     metadata_df.to_dict(orient='records')}

    with open("artifacts/output/partitions.json", 'w+') as f:
        json.dump(metadata_dict, f, indent=2)

    data_metadata = {
        "num_rows": int(metadata_df["num_rows"].sum())
    }

    with open("artifacts/output/data_stats.json", 'w+') as f:
        json.dump(data_metadata, f, indent=2)

    print("Finished saving artefacts")
