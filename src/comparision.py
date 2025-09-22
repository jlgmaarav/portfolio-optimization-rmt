import pandas as pd
import sys
import logging
import os
from typing import Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_asset_count(file1: str, file2: str) -> Tuple[bool, int, int]:
    """Compare the number of columns (assets) in two CSV files.

    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.

    Returns:
        Tuple[bool, int, int]: (True if counts match, count1, count2)

    Examples:
        >>> match, count1, count2 = compare_asset_count('data.csv', 'data2.csv')
        >>> print(match)
        True
    """
    if not os.path.exists(file1):
        logger.error(f"File '{file1}' does not exist.")
        sys.exit(1)
    if not os.path.exists(file2):
        logger.error(f"File '{file2}' does not exist.")
        sys.exit(1)

    logger.info(f"Comparing the number of assets in '{file1}' and '{file2}'...")

    try:
        df1 = pd.read_csv(file1).astype(np.float32)
        logger.info(f"'{file1}' loaded successfully with {len(df1.columns)} assets.")
    except Exception as e:
        logger.error(f"Error loading '{file1}': {str(e)}")
        sys.exit(1)

    try:
        df2 = pd.read_csv(file2).astype(np.float32)
        logger.info(f"'{file2}' loaded successfully with {len(df2.columns)} assets.")
    except Exception as e:
        logger.error(f"Error loading '{file2}': {str(e)}")
        sys.exit(1)

    count1 = len(df1.columns)
    count2 = len(df2.columns)

    if count1 == count2:
        logger.info("Success! The number of assets in both files is the same.")
        return True, count1, count2
    else:
        logger.warning(f"Attention! Asset counts differ: '{file1}' has {count1}, '{file2}' has {count2}.")
        return False, count1, count2

if __name__ == "__main__":
    compare_asset_count('data.csv', 'data2.csv')