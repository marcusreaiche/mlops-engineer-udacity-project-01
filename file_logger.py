"""
Logger used in the churn_script_logging_and_tests module

Author: Marcus Reaiche
Sep 7, 2023
"""
import logging

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
file_logger = logging.getLogger(__name__)
