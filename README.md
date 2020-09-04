# Models

This repository contains demo models for the Teradata AOA. See the [HowTo](./HOWTO.md) guide for a brief tutorial on how to add new models.

# Available Models

More human friendly folder names for each model under model_definitions is tracked in [issue-105](https://github.com/ThinkBigAnalytics/AoaCoreService/issues/105).

- [Tensorflow Sentitment Analysis](model_definitions/74eca506-e967-48f1-92ad-fb217b07e181)
- [Python XGboost (Diabetes)](model_definitions/03c9a01f-bd46-4e7c-9a60-4282039094e6)
- [Python MLE XGboost (Diabetes)](model_definitions/920ebf0e-1f0e-442a-94d1-214f63b8b820)
- [SQL MLE XGboost (Diabetes)](model_definitions/d0cfc94e-fa8d-4b08-a9ff-341ef2739adf)
- [R GBM (Diabetes)](model_definitions/bf6a52b2-b595-4358-ac4f-24fb41a85c45)
- [PySpark Logistic Regression](model_definitions/149e31ed-c554-46b4-95d2-00c5c43320fb)


# Adding new Models

Use the aoa cli (see [here](https://pypi.org/project/aoa/))

```
aoa add
```

## Running Models Locally

Use the aoa cli (see [here](https://pypi.org/project/aoa/))

```
aoa run -h
```
    
## Detailed Documentation

To see more detailed documentation on supported model signatures, additional variables passed to the models, how we store and retrieve artifacts and metrics, see the following [document](https://github.com/ThinkBigAnalytics/AoaCoreService/blob/master/docs/ModelsSupport.md).