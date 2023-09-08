"""
Implement unit tests for churn_library module

Author: Marcus Reaiche
Sep 8, 2023
"""

import os
import numpy as np
import pytest
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from file_logger import file_logger
from constants import (
    DATA_FILEPATH,
    IMG_FILE_EXT,
    CATEGORICAL_COLS,
    RESPONSE_COL,
    TEST_SIZE,
    RANDOM_STATE,
    FEATURES_COLS)
import churn_library
from churn_library import (
    import_data,
    perform_eda,
    perform_feature_engineering,
    encoder_helper,
    train_models)


# Pytest fixtures
# Use kwarg name in @pytest.fixture to avoid pylint complaints
# See stackoverflow and search for:
# /questions/46089480/pytest-fixtures-redefining-name-from-outer-scope-pylint
@pytest.fixture(scope='module', name='data_path')
def fixture_data_path():
    """data_path parameter for import_data"""
    return DATA_FILEPATH


@pytest.fixture(scope='module', name='data_before_eda')
def fixture_data_before_eda(data_path):
    """data before EDA is done"""
    return import_data(data_path)


@pytest.fixture(scope='module', name='data_after_eda')
def fixture_data_after_eda(data_before_eda):
    """data after EDA is done"""
    return (
        data_before_eda
        .assign(Churn=lambda df: np.where(
            df.Attrition_Flag == 'Existing Customer',
            0,
            1)))


@pytest.fixture(scope='module', name='data_encoded')
def fixture_data_encoded(data_after_eda):
    """data after categorical variables are encoded"""
    data = encoder_helper(data_after_eda, CATEGORICAL_COLS)
    return data


@pytest.fixture(scope='module', name='features')
def fixture_features(data_encoded):
    """data features or X"""
    return data_encoded.loc[:, FEATURES_COLS]


@pytest.fixture(scope='module', name='target')
def fixtures_target(data_encoded):
    """data target or y"""
    return data_encoded.loc[:, RESPONSE_COL]


@pytest.fixture(scope='module', name='data_split')
def fixture_data_split(features, target):
    """dictionary containing the data split in training and testing"""
    x_train, x_test, y_train, y_test = \
            train_test_split(features, target,
                             test_size=TEST_SIZE,
                             random_state=RANDOM_STATE)
    return dict(x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test)


def test_import_data(data_path):
    '''
    Test import data
    '''
    file_logger.info("Testing import_data: START")
    try:
        data = import_data(data_path)
    except FileNotFoundError as err:
        file_logger.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        log_msg = ("Testing import_data:" +
                   " The file doesn't appear to have rows and columns")
        file_logger.error(log_msg)
        raise err
    file_logger.info("Testing import_data: SUCCESS")


def test_perform_eda(data_before_eda, tmp_path, monkeypatch):
    '''
    Test perform_eda function
    '''
    file_logger.info('Testing perform_eda - START')
    # Set temporary directory to save EDA files
    tmp_images_eda_directory = tmp_path / 'images' / 'eda'
    tmp_images_eda_directory.mkdir(parents=True)
    monkeypatch.setattr(churn_library, "IMG_EDA_DIR", tmp_images_eda_directory)
    try:
        perform_eda(data_before_eda)
        # Test that 'Churn' columns was created
        assert 'Churn' in data_before_eda
        file_logger.info('"Churn" column was created')
    except AssertionError as err:
        file_logger.error('"Churn" column was not created')
        raise err
    try:
        # Test that 'Churn' column has only zeros and ones
        assert set(data_before_eda.Churn.value_counts().index).issubset({0, 1})
        file_logger.info('"Churn" column has only 0\'s and 1\'s')
    except AssertionError as err:
        file_logger.error('"Churn" column has values not in {0, 1}')
        raise err
    try:
        # Check that the five images were saved in the temporary directory
        expected_images = {
            'total_transaction_distribution.png',
            'customer_age_distribution.png',
            'churn_distribution.png',
            'marital_status_distribution.png',
            'heatmap.png'}
        saved_images = {
            file for file in os.listdir(tmp_images_eda_directory)
            if file.endswith(IMG_FILE_EXT)}
        assert saved_images == expected_images
        file_logger.info('Expected image files were saved to disk')
    except AssertionError as err:
        file_logger.error('Saved image files do not agree with expected images')
        raise err
    file_logger.info('Testing perform_eda - SUCCESS')


def test_encoder_helper(data_after_eda):
    '''
    Test encoder helper
    '''
    file_logger.info('Test encoder_helper - START')
    try:
        data = encoder_helper(data_after_eda, CATEGORICAL_COLS)
        expected_cols = [col + '_' + RESPONSE_COL for col in CATEGORICAL_COLS]
        assert set(expected_cols).issubset(data.columns)
        log_msg = f'Categorical cols {expected_cols} were created'
        file_logger.info(log_msg)
    except AssertionError as err:
        file_logger.error('Some categorical cols were not created')
        raise err
    file_logger.info('Test encoder_helper - SUCCESS')


def test_perform_feature_engineering(data_after_eda, data_split):
    '''
    Test perform_feature_engineering
    '''
    file_logger.info('Testing perform_feature_engineering - START')
    try:
        x_train, x_test, y_train, y_test = \
            perform_feature_engineering(data_after_eda)
        assert (x_train.equals(data_split["x_train"]) and
                x_test.equals(data_split["x_test"]) and
                y_train.equals(data_split["y_train"]) and
                y_test.equals(data_split["y_test"]))
    except AssertionError as err:
        file_logger.error('Split data do not agree')
        raise err
    file_logger.info('Test perform_feature_engineering - SUCCESS')


def test_train_models(data_split, tmp_path, monkeypatch):
    '''
    Test train_models
    '''
    file_logger.info('Testing train_models - START')
    # Set temporary paths to save models
    tmp_models_directory = tmp_path / 'models'
    tmp_lrc_model_filepath = tmp_models_directory / 'logistic_model.pkl'
    tmp_rfc_model_filepath = tmp_models_directory / 'rfc_model.pkl'
    tmp_models_directory.mkdir(parents=True)
    # Use monkeypatch to reset global variables during testing
    monkeypatch.setattr(churn_library,
                        "LRC_MODEL_FILEPATH",
                        tmp_lrc_model_filepath)
    monkeypatch.setattr(churn_library,
                        "RFC_MODEL_FILEPATH",
                        tmp_rfc_model_filepath)

    # Set temporary results directory
    tmp_results_directory = tmp_path / 'images' / 'results'
    tmp_roc_curve_filepath = tmp_results_directory / 'roc_curve_result.png'
    tmp_img_lrc_filepath = tmp_results_directory / 'logistic_results.png'
    tmp_img_rfc_filepath = tmp_results_directory / 'rf_results.png'
    tmp_feature_importances_filepath = \
        tmp_results_directory / 'feature_importances.png'
    tmp_results_directory.mkdir(parents=True)
    # Use monkeypatch to reset global variables during testing
    monkeypatch.setattr(churn_library,
                        "IMG_LRC_FILEPATH",
                        tmp_img_lrc_filepath)
    monkeypatch.setattr(churn_library,
                        "IMG_RFC_FILEPATH",
                        tmp_img_rfc_filepath)
    monkeypatch.setattr(churn_library,
                        "FEATURE_IMPORTANCES_FILEPATH",
                        tmp_feature_importances_filepath)
    monkeypatch.setattr(churn_library,
                        "ROC_CURVE_FILEPATH",
                        tmp_roc_curve_filepath)
    # Set params grid to singleton (speed up test execution)
    param_grid = {
        'n_estimators': [200],
        'max_features': ['auto'],
        'max_depth' : [100],
        'criterion' :['entropy']}
    monkeypatch.setattr(churn_library, "RFC_PARAM_GRID", param_grid)
    try:
        train_models(data_split['x_train'],
                     data_split['x_test'],
                     data_split['y_train'],
                     data_split['y_test'])
        # Check that models directory has 2 models
        model_files = {file for file in os.listdir(tmp_models_directory)
                       if file.endswith('.pkl')}
        expected_model_files = {'logistic_model.pkl', 'rfc_model.pkl'}
        assert model_files == expected_model_files
        file_logger.info('Model files - OKAY')
    except AssertionError as err:
        file_logger.error('Model files do not agree with expected results')
        raise err

    try:
        # Check that models are of the class LogisticRegression and
        # RandomForestClassifier
        with open(tmp_lrc_model_filepath, 'rb') as file:
            lrc_model = joblib.load(file)
        assert isinstance(lrc_model, LogisticRegression)

        with open(tmp_rfc_model_filepath, 'rb') as file:
            rfc_model = joblib.load(file)
        assert isinstance(rfc_model, RandomForestClassifier)
        file_logger.info('Models instances - OKAY')
    except AssertionError as err:
        file_logger.error('Models instances are of unexpected class')
        raise err

    try:
        # Check that expected images are in tmp_results_directory
        expected_img_in_results_directory = {
            'feature_importances.png',
            'logistic_results.png',
            'rf_results.png',
            'roc_curve_result.png'
        }
        imgs_in_results_directory = {
            file for file in os.listdir(tmp_results_directory)
            if file.endswith(IMG_FILE_EXT)}
        assert expected_img_in_results_directory == imgs_in_results_directory
        file_logger.info('Images in images/results - OKAY')
    except AssertionError as err:
        file_logger.error('Unexpected images in results')
        raise err
    file_logger.info("Testing train_models - SUCCESS")


if __name__ == "__main__":
    pytest.main([__file__])
