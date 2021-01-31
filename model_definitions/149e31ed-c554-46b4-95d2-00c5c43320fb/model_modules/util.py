import urllib.request


def read_dataframe(spark, url):
    # in a real world scenario, you would read from S3, HDFS, Teradata,
    # etc but for demo reading from url. we could read via pandas.read_csv but just to show pyspark ...
    urllib.request.urlretrieve(url, "/tmp/data.csv")
    column_names = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes"]

    return spark.read.format("csv") \
        .option("inferSchema", "true") \
        .load("/tmp/data.csv") \
        .toDF(*column_names)
