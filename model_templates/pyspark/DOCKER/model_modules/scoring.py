

def evaluate(spark, data_conf, model_conf, **kwargs):
    """pySpark evaluate method called by AOA framework

    Parameters:
    spark (SparkSession): The SparkSession created by the framework
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    # dump results as json file evaluation.json to models/ folder
    print("Evaluation complete...")
