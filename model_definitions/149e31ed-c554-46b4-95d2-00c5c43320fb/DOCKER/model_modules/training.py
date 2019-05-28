from pyspark.ml.classification import LogisticRegression


def train(spark, data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    # Load training data
    training = spark.read.format("libsvm").load(data_conf["data_path"])

    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(training)

    # Print the coefficients and intercept for logistic regression
    print("Coefficients: " + str(lrModel.coefficients))
    print("Intercept: " + str(lrModel.intercept))

    # We can also use the multinomial family for binary classification
    mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

    # Fit the model
    mlrModel = mlr.fit(training)

    # Print the coefficients and intercepts for logistic regression with multinomial family
    print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
    print("Multinomial intercepts: " + str(mlrModel.interceptVector))

    print("Starting training...")

    # fit model to training data

    print("Finished training")

    # export model artefacts to models/ folder

    print("Saved trained model")