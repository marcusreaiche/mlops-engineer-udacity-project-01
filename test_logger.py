"""
Logger used in the churn_script_logging_and_tests module

Author: Marcus Reaiche
Sep 7, 2023
"""
import logging
from constants import LOGS_DIR, LOGS_FILEPATH
from helpers import create_dir

# Create LOGS_DIR if directory does not exist
create_dir(LOGS_DIR)

# Configure logging
logging.basicConfig(
    filename=LOGS_FILEPATH,
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
test_logger = logging.getLogger(__name__)
