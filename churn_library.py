# library doc string
"""
This module completes the process for solving the data science process including:

 - EDA
 - Feature Engineering (including encoding of categorical variables)
 - Model Training
 - Prediction
 - Model Evaluation

Author: Marcus Reaiche
Sep 7, 2023
"""

# import libraries
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from constants import (
    DATA_FILEPATH,
    IMG_EDA_DIR,
    IMG_LRC_FILEPATH,
    IMG_RFC_FILEPATH,
    CATEGORICAL_COLS,
    RESPONSE_COL,
    FEATURES_COLS,
    RANDOM_STATE,
    TEST_SIZE,
    LRC_SOLVER,
    LRC_MAX_ITER,
    LRC_MODEL_FILEPATH,
    RFC_PARAM_GRID,
    RFC_CV,
    RFC_MODEL_FILEPATH,
    FEATURE_IMPORTANCES_FILEPATH)
from helpers import (
    create_eda_figs,
    save_figs,
    _build_classification_report_image,
    generate_roc_curves,
    save_model)


os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# Logger basic config
# - messages are logged to stdout
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


def perform_eda(data):
    '''
    Perform eda on data and save figures to images folder
    input:
            data: pandas dataframe

    output:
            None
    '''
    # Perfom EDA
    logging.info('data.head')
    logging.info(data.head())
    logging.info('data.shape')
    logging.info(data.shape)
    logging.info('Checking for nulls')
    logging.info(data.isnull().sum())
    logging.info('data.describe')
    logging.info(data.describe())
    # Create Churn col
    data['Churn'] = (data['Attrition_Flag']
                   .apply(lambda val: 0 if val == "Existing Customer" else 1))
    # Create EDA figures
    figs_dict = create_eda_figs(data)
    # Save EDA figures
    logging.info('Saving EDA figures to disk')
    save_figs(figs_dict=figs_dict, fig_dir=IMG_EDA_DIR)


def encoder_helper(data, category_lst, response=RESPONSE_COL):
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 16 from the notebook

    input:
            data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
                      used for naming variables or index y column]

    output:
            data: pandas dataframe with new columns for
    '''
    for cat in category_lst:
        cat_group = data.groupby(cat)[response].mean()
        data[cat + '_' + response] = data[cat].map(cat_group)
    return data


def perform_feature_engineering(data, response=RESPONSE_COL):
    '''
    input:
              data: pandas dataframe
              response: string of response name [optional argument that could be
                        used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Create encoded columns
    encoder_helper(data, category_lst=CATEGORICAL_COLS, response=response)
    # Set X and y
    features = data.loc[:, FEATURES_COLS]
    target = data[response]
    # Split data into training and test sets
    return train_test_split(features, target,
                            test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    Produces classification report for training and testing results and stores
    report as image in images folder
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
    # Logistic Regression
    lrc_fig = _build_classification_report_image(y_train,
                                       y_test,
                                       y_train_preds_lr,
                                       y_test_preds_lr,
                                       "Logistic Regression")
    # Save figure to disk
    lrc_fig.savefig(IMG_LRC_FILEPATH)
    # Random Forest
    rfc_fig = _build_classification_report_image(y_train,
                                       y_test,
                                       y_train_preds_rf,
                                       y_test_preds_rf,
                                       "Random Forest")
    rfc_fig.savefig(IMG_RFC_FILEPATH)



def feature_importance_plot(model, features, output_pth):
    '''
    Creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = features.columns[indices].to_list()
    # Create plot
    fig = plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(features.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(features.shape[1]), names, rotation=90)
    # Ensure plot is within figure
    plt.tight_layout()
    # Save figure to disk
    fig.savefig(output_pth)


def train_models(features_train, features_test, target_train, target_test):
    '''
    Train, store model results: images + scores, and store models
    input:
              features_train: features training data
              features_test: features testing data
              target_train: target training data
              target_test: target testing data
    output:
              None
    '''
    # Logistic regression model
    logging.info('Train LogisticRegressionClassifier')
    lrc = LogisticRegression(solver=LRC_SOLVER, max_iter=LRC_MAX_ITER)
    lrc.fit(features_train, target_train)
    y_train_preds_lr = lrc.predict(features_train)
    y_test_preds_lr = lrc.predict(features_test)
    # Random Forest model
    logging.info('Train RandomForestClassifier (GridSearch)')
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=RFC_PARAM_GRID, cv=RFC_CV)
    cv_rfc.fit(features_train, target_train)
    best_rfc = cv_rfc.best_estimator_
    y_train_preds_rf = best_rfc.predict(features_train)
    y_test_preds_rf = best_rfc.predict(features_test)
    # Save models to disk
    logging.info('Save models to disk')
    save_model(lrc, LRC_MODEL_FILEPATH)
    save_model(best_rfc, RFC_MODEL_FILEPATH)
    # Generate ROC curves
    logging.info('Generate ROC curves')
    generate_roc_curves([best_rfc, lrc], features_test, target_test)
    # Generate classification reports images
    logging.info('Generate classification report images')
    classification_report_image(target_train,
                                target_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    # Generate feature importances plot
    logging.info('Generate feature importances plot')
    features = pd.concat([features_train, features_test])
    feature_importance_plot(best_rfc, features, FEATURE_IMPORTANCES_FILEPATH)


if __name__ == '__main__':
    logging.info('Import data')
    data = import_data(pth=DATA_FILEPATH)
    logging.info('Perform EDA')
    perform_eda(data)
    logging.info('Perform feature engineering')
    features_train, features_test, target_train, target_test = \
        perform_feature_engineering(data)
    logging.info('Training models and saving models and plots to disk')
    train_models(features_train, features_test, target_train, target_test)
