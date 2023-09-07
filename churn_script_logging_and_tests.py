import pytest
from file_logger import file_logger
from constants import DATA_FILEPATH
from churn_library import (
	import_data,
	perform_eda,
	perform_feature_engineering,
	encoder_helper,
	train_models)

# Pytest fixtures
@pytest.fixture(scope="module")
def data_path():
	return DATA_FILEPATH


def test_import_data(data_path):
	'''
	Test import data
	'''
	try:
		data = import_data(data_path)
		file_logger.info("Testing import_data: SUCCESS")
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


def test_perform_eda():
	'''
	Test perform eda function
	'''
	raise NotImplementedError('Implement test_perform_eda')


def test_encoder_helper():
	'''
	Test encoder helper
	'''
	raise NotImplementedError('Implement test_encoder_helper')


def test_perform_feature_engineering():
	'''
	Test perform_feature_engineering
	'''
	raise NotImplementedError('Implement test_perform_feature_engineering')


def test_train_models():
	'''
	Test train_models
	'''
	raise NotImplementedError('Implement test_train_models')


if __name__ == "__main__":
	pytest.main([__file__])
