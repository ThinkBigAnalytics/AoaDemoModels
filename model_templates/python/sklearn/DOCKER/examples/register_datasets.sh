#!/bin/bash

curl -X POST -u admin:admin \
    -H "Content-Type: application/json" \
    -H "AOA-Project-ID: 23e1df4b-b630-47a1-ab80-7ad5385fcd8d" \
    -d '{"name":"Iris Training Dataset 1","metadata":{"data_table":"./model_modules/data/iris_train1.csv"}}' \
    http://localhost:8080/api/datasets

curl -X POST -u admin:admin \
    -H "Content-Type: application/json" \
    -H "AOA-Project-ID: 23e1df4b-b630-47a1-ab80-7ad5385fcd8d" \
    -d '{"name":"Iris Training Dataset 2","metadata":{"data_table":"./model_modules/data/iris_train2.csv"}}' \
    http://localhost:8080/api/datasets

curl -X POST -u admin:admin \
    -H "Content-Type: application/json" \
    -H "AOA-Project-ID: 23e1df4b-b630-47a1-ab80-7ad5385fcd8d" \
    -d '{"name":"Iris Evaluation Dataset","metadata":{"data_table":"./model_modules/data/iris_evaluate.csv"}}' \
    http://localhost:8080/api/datasets