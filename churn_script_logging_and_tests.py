import os
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
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
@pytest.fixture(scope='module')
def data_path():
    return DATA_FILEPATH


@pytest.fixture(scope='module')
def data_before_eda(data_path):
    return import_data(data_path)


@pytest.fixture(scope='module')
def data_after_eda(data_before_eda):
    return (
        data_before_eda
        .assign(Churn=lambda df: np.where(
            df.Attrition_Flag == 'Existing Customer',
            0,
            1)))


@pytest.fixture(scope='module')
def data_encoded(data_after_eda):
    data = encoder_helper(data_after_eda, CATEGORICAL_COLS)
    return data


@pytest.fixture(scope='module')
def features(data_encoded):
    return data_encoded.loc[:, FEATURES_COLS]


@pytest.fixture(scope='module')
def target(data_encoded):
    return data_encoded.loc[:, RESPONSE_COL]


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
        file_logger.error("Testing import_data:" +
                          " The file doesn't appear to have rows and columns")
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
        expected_images = set([
            'total_transaction_distribution.png',
            'customer_age_distribution.png',
            'churn_distribution.png',
            'marital_status_distribution.png',
            'heatmap.png'])
        saved_images = set([
            file for file in os.listdir(tmp_images_eda_directory)
            if file.endswith(IMG_FILE_EXT)])
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
        file_logger.info(f'Categorical cols {expected_cols} were created')
    except AssertionError as err:
        file_logger.error('Some categorical cols were not created')
        raise err
    file_logger.info('Test encoder_helper - SUCCESS')


def test_perform_feature_engineering(data_after_eda, features, target):
    '''
    Test perform_feature_engineering
    '''
    file_logger.info('Testing perform_feature_engineering - START')
    try:
        x_train, x_test, y_train, y_test = \
            perform_feature_engineering(data_after_eda)
        x_train_expected, x_test_expected, y_train_expected, y_test_expected = \
            train_test_split(features, target,
                             test_size=TEST_SIZE,
                             random_state=RANDOM_STATE)
        assert (x_train.equals(x_train_expected) and
                x_test.equals(x_test_expected) and
                y_train.equals(y_train_expected) and
                y_test.equals(y_test_expected))
    except AssertionError as err:
        file_logger.error('Split data do not agree')
        raise err
    file_logger.info('Test perform_feature_engineering - SUCCESS')


def test_train_models():
    '''
    Test train_models
    '''
    raise NotImplementedError('Implement test_train_models')


if __name__ == "__main__":
    pytest.main([__file__])
