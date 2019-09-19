
# Overview
Sample pyspark model using spark mlib LogisticRegressionModel algorithm.

# Datasets
The dataset we're using is the standard libsvm dataset from spark. To prevent local path issues we download this file from a url and then load it into spark. We also set the test / train split.
 
*Temporarily with spark jobs we need to store the model in some path where it can be later moved to the correct model artefact repository location. As spark is distributed, this must be a distributed store and so we can't ask it to store them directly in the "models" folder where the driver is running. In the example below it is local as in the demo we are runnning spark embedded mode but in a real scenario this must be a distributed store like hdfs or s3.*

    data_conf = {
        "url": "https://raw.githubusercontent.com/apache/spark/branch-2.4/data/mllib/sample_libsvm_data.txt"
    }

for evaluation (and scoring) - note the url and test_split is only for this demo model and in a normal scenario you would have a different evaluation dataset completely

    data_conf = {
        "url": "https://raw.githubusercontent.com/apache/spark/branch-2.4/data/mllib/sample_libsvm_data.txt",
        "predictions": "<predictions-output>"
    }

# Training
The [training.py](model_modules/training.py) produces the following artifacts in the artefact repository

    lr/data/part-*****          (partition files for the lreg model parameters)
    lr/metadata/part-00000      (metadata file for the lreg model structure)


# Evaluation
Evaluation is also performed in [scoring.evaluate](model_modules/scoring.py) and it returns the following metrics

    roc: <roc>
    

# Scoring 
We could put this model behind a RESTful API by exporting the model as PMML or storing as a Java POJO with spark mlib. These could then be deployed in the PMML RESTful engine we support. If it is a s a batch or streaming process, you will need to create the relevant engine/job to run it. 
