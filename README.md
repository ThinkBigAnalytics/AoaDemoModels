

# Models

This repository contains the example Diabetes Prediction model code for ModelOps. We provide a number of demo projects and associated repositories. The goal of these examples is to provide a simple reference implementation for users and not to provide a detailed data science example for each use case. 

Please refer to the [Official Documentation](https://docs.teradata.com/r/Teradata-VantageTM-ModelOps-User-Guide/June-2022) for more information.

## Available Models

####  Git

`Git` models are those models where we manage all the code for training evaluation and scoring. We provide notebooks for the Python example.

- [Python Diabetes Prediction](model_definitions/python-diabetes)
- [R Diabetes Prediction](model_definitions/r-diabetes)

#### BYOM

`BYOM` is related to [Teradata BYOM](https://docs.teradata.com/r/Teradata-VantageTM-Bring-Your-Own-Model-User-Guide/May-2022/Bring-Your-Own-Model) where we use an open model format such as `PMML` or `ONNX`. In this example, we provide a notebook to produce the `PMML` model artefact along with examples for how to query it dynamically in Teradata. 

- [Diabetes Prediction](byom/pima)


