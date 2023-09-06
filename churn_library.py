# library doc string


# import libraries
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from constants import (
    IMG_EDA_DIR,
    CATEGORICAL_COLS,
    RESPONSE_COL,
    FEATURES_COLS)
from helpers import (
    create_eda_figs,
    save_figs)


os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(
    level = logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    Returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(df):
    '''
    Perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Perfom EDA
    logging.info('df.head')
    logging.info(df.head())
    logging.info('df.shape')
    logging.info(df.shape)
    logging.info('Checking for nulls')
    logging.info(df.isnull().sum())
    logging.info('df.describe')
    logging.info(df.describe())
    # Create Churn col
    df['Churn'] = (df['Attrition_Flag']
                   .apply(lambda val: 0 if val == "Existing Customer" else 1))
    # Create EDA figures
    figs_dict = create_eda_figs(df)
    # Save EDA figures
    save_figs(figs_dict=figs_dict, fig_dir=IMG_EDA_DIR)


def encoder_helper(df, category_lst, response=RESPONSE_COL):
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 16 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for cat in category_lst:
        cat_group = df.groupby(cat)[response].mean()
        df[cat + '_' + response] = df[cat].map(cat_group)
    return df


def perform_feature_engineering(df, response=RESPONSE_COL):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Create encoded columns
    encoder_helper(df, category_lst=CATEGORICAL_COLS, response=response)
    # Set X and y
    X = df.loc[:, FEATURES_COLS]
    y = df[RESPONSE_COL]
    # Split data into training and test sets
    return train_test_split(X, y, test_size=0.3, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    Produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    Creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    Train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass
