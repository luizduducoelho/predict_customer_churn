# Predict Customer Churn

- Project **Predict Customer Churn** using ML DevOps. 

## Project Description
Project for Client Churn Prediction.
ML ops techniques are used to convert the jupyter notebook to scrits. 
We also employ logging and testing. 

## Files and data description
Overview of the files and data present in the root directory: 
```
project
│   README.md
│   requirements_py3.6.txt    
│   requirements_py3.8.txt
│   .gitattributes    
│   Guide.ipynb: initial file   
│   churn_notebook.ipynb: initial file   
|   churn_library.py: main script
│   churn_script_logging_and_tests.py: testing script
│   conftest.py: configure testing variables
│
└───images
│   │   
│   │
│   └───eda
│   |    │   Images for exploratory data analisys
│   |    │   ...
|   | 
|   |
│   └───results
│       │   Feature importance plots and roc curves
│       │   ...
│   
└───Models
|    │   Models binaries
│   
|
└───Data
|    │   Csv file with dataset
|    
│   
└───Logs
    │   Logging file
    
```

## Running Files
How do you run your files? What should happen when you run your files?
1. First install necessary packages by running 
   ```
    python -m pip install -r requirements_py3.8.txt
    ```
2. Run training with the command below. The images with results as well as the model binaries will be in their respective folders.
   ```
    python churn_library.py
    ```
   
3. Run tests with the following command. Check logs in `logs/churn_library.logs` file. 
   ```
    python churn_script_logging_and_tests.py
    ```
