# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

In this project, we identify credit card customers that are most likely to churn. The basic problem that we aim to solve is:

> How do we identify (and later intervene with) customers who are likely to churn?

The data set used on this Project was pulled from [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code). 

The Project's code is the result of refactoring a [Jupyter Notebook](https://github.com/marcusreaiche/mlops-engineer-udacity-project-01/blob/main/churn_notebook.ipynb) applying engineering best practices for implementing software (modular, documented, and tested). The package have the flexibility of being run interactively or from the command-line interface (CLI).

## Dependencies

For Pyton 3.8

```txt
scikit-learn==0.24.1
shap==0.40.0
joblib==1.0.1
pandas==1.2.4
numpy==1.20.1
matplotlib==3.3.4
seaborn==0.11.2
pylint==2.7.4
autopep8==1.5.6
pytest==7.4.2
notebook==7.0.3
```

There are a number of ways to install the correct Python version and to create a virtual environment. 

Using miniconda
- Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-other-installer-links.html)
- After configuring the shell to run miniconda, create the `customer_churn` environment:

```bash
$ conda create -n customer_churn python=3.8
```

Activate the environment:

```bash
$ conda activate customer_churn
```

Install the packages

```bash
$ pip install -r requirements_py3.8.txt
```


## Files and data description
The Project is composed of the following files:

### churn_library.py

The churn_library.py is a module of functions to find customers who are likely to churn. This module can also be executed as a script using the CLI.

All the data science tasks are performed in this module such as:
- EDA
- Feature engineering
- Model training

### constants.py
In the constants.py module, all constants used along the Project are set.
### helpers.py
Auxiliary functions used in churn_library module are implemented.

### churn_script_logging_and_tests.py
Defines unit tests using pytest and generates ./log/churn_library.log when executed from the CLI.
### test_logger.py
Defines logger used in churn_script_logging_and_tests.
## Running Files
### Running the `churn_library` script

Execute in the CLI the following command:

```bash
$ python churn_library.py
```
This command will perform the following tasks:
- Load data set in memory
- Perform EDA
    - routine log EDA results to stdout
    - routine produces images in ./images/eda
- Perform feature engineering to increase model performance
- train classification models
    - models tested:
        - logistic regression, and
        - random forest classifier (in a grid search)
    - save models to ./models
    - generate ROC curves for both models
        - image is saved in ./images/results
    - generate classification reports for logistic and best random forest classifier model
        - reports are saved in ./images/results
    - generate feature importances plot for best random forest model
        - image is saved in ./images/results


### Running the `churn_script_logging_and_test` script

Execute in the CLI the following command:

```bash
$ python churn_script_logging_and_test.py
```
This command performs the unit tests defined in the script and will produce the log file located in ./log/churn_library.log
