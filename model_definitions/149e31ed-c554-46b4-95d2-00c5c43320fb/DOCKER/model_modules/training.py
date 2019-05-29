from pyspark.ml.classification import LogisticRegression


def train(spark, data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    # Load training data - standard spark dataset data/mllib/sample_libsvm_data.txt
    training = spark.read.format("libsvm").load(data_conf["data_path"])

    lr = LogisticRegression(maxIter=hyperparams["maxIter"],
                            regParam=hyperparams["regParam"],
                            elasticNetParam=hyperparams["elasticNetParam"])

    print("Starting training...")

    lr_model = lr.fit(training)

    # Print the coefficients and intercept for logistic regression
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

    print("Finished training")

    # export model artefacts to models/ folder

    print("Saved trained model")
    lr_model.save("file:///models")