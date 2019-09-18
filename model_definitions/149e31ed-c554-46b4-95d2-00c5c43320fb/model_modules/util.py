import urllib.request


def read_dataset_from_url(spark, url):
    # for this demo we're downloading the dataset locally and then reading it. This is obviously not production setting
    # https://raw.githubusercontent.com/apache/spark/branch-2.4/data/mllib/sample_libsvm_data.txt
    urllib.request.urlretrieve(url, "/tmp/data.txt")
    return spark.read.format("libsvm").load("/tmp/data.txt")