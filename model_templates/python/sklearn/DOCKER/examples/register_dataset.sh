#!/bin/bash

curl -X POST -u admin:admin \
    -H "Content-Type: application/json" \
    -H "AOA-Project-ID: 23e1df4b-b630-47a1-ab80-7ad5385fcd8d" \
    -d '{"name":"Iris Training Dataset","metadata":{"location":"https://datahub.io/machine-learning/iris/r/iris.csv"}}' \
    http://localhost:8080/api/datasets
