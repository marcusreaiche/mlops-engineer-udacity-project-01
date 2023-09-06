import os
from os.path import join as joinpath


CATEGORICAL_COLS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

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
    'Avg_Utilization_Ratio'
]

# Images constants
CURR_FILE_DIR = os.path.dirname(__file__)
IMG_DIR = joinpath(CURR_FILE_DIR, 'images')
IMG_EDA_DIR = joinpath(IMG_DIR, 'eda')
IMG_RESULTS_DIR = joinpath(IMG_DIR, 'results')
IMG_SIZE = (20, 10)
IMG_FILE_EXT = 'png'
