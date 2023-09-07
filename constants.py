import os
from os.path import join as joinpath

# Lists of Columns
CATEGORICAL_COLS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']
QUANTITATIVE_COLS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio']
FEATURES_COLS = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn']
RESPONSE_COL = 'Churn'

# Directory and file paths
CURR_FILE_DIR = os.path.dirname(__file__)
IMG_DIR = joinpath(CURR_FILE_DIR, 'images')
IMG_EDA_DIR = joinpath(IMG_DIR, 'eda')
IMG_RESULTS_DIR = joinpath(IMG_DIR, 'results')
ROC_CURVE_FILEPATH = joinpath(IMG_RESULTS_DIR, 'roc_curve_result.png')
FEATURE_IMPORTANCES_FILEPATH = joinpath(IMG_RESULTS_DIR,
                                        'feature_importances.png')
MODELS_DIR = joinpath(CURR_FILE_DIR, 'models')
LRC_MODEL_FILEPATH = joinpath(MODELS_DIR, 'logistic_model.pkl')
RFC_MODEL_FILEPATH = joinpath(MODELS_DIR, 'rfc_model.pkl')

# Images constants
IMG_EDA_SIZE = (20, 10)
IMG_ROC_CURVES_SIZE = (15, 8)
IMG_CLASSIFICATION_REPORT_SIZE = (5, 5)
IMG_FILE_EXT = 'png'

# Train - Test - Split params
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Grid Search Hyperparameters for Random Forest
RFC_PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']}
RFC_CV = 5

# Logistic Regression Hyperparameters
LRC_SOLVER = 'lbfgs'
LRC_MAX_ITER = 3_000
