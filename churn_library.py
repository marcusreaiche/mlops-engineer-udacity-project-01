# library doc string


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
    IMG_EDA_DIR,
    IMG_RESULTS_DIR,
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
    RFC_MODEL_FILEPATH)
from helpers import (
    create_eda_figs,
    save_figs,
    _build_classification_report_image,
    generate_roc_curves,
    save_model)


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
    return train_test_split(X, y,
                            test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)


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
    # Logistic Regression
    lr_filepath = os.path.join(IMG_RESULTS_DIR, 'logistic_results.png')
    _build_classification_report_image(y_train,
                                       y_test,
                                       y_train_preds_lr,
                                       y_test_preds_lr,
                                       "Logistic Regression",
                                       lr_filepath)
    # Random Forest
    rf_filepath = os.path.join(IMG_RESULTS_DIR, 'rf_results.png')
    _build_classification_report_image(y_train,
                                       y_test,
                                       y_train_preds_rf,
                                       y_test_preds_rf,
                                       "Random Forest",
                                       rf_filepath)


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
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = X_data.columns[indices].to_list()
    # Create plot
    fig = plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    # Ensure plot is within figure
    plt.tight_layout()
    # Save figure to disk
    fig.savefig(output_pth)


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
    # Logistic regression model
    logging.info('Train LogisticRegressionClassifier')
    lrc = LogisticRegression(solver=LRC_SOLVER, max_iter=LRC_MAX_ITER)
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    # Random Forest model
    logging.info('Train RandomForestClassifier (GridSearch)')
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=RFC_PARAM_GRID, cv=RFC_CV)
    cv_rfc.fit(X_train, y_train)
    best_rfc = cv_rfc.best_estimator_
    y_train_preds_rf = best_rfc.predict(X_train)
    y_test_preds_rf = best_rfc.predict(X_test)
    # Save models to disk
    logging.info('Save models to disk')
    save_model(lrc, LRC_MODEL_FILEPATH)
    save_model(best_rfc, RFC_MODEL_FILEPATH)
    # Generate ROC curves
    logging.info('Generate ROC curves')
    generate_roc_curves([best_rfc, lrc], X_test, y_test)
    # Generate classification reports images
    logging.info('Generate classification report images')
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
