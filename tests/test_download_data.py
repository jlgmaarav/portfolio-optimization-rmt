import unittest
import pandas as pd
import os
from datetime import datetime
from src.download_data import download_and_save, download_batch

class TestDownloadData(unittest.TestCase):
    def setUp(self):
        """Set up test parameters."""
        self.tickers = ['AAPL', 'MSFT']
        self.start = '2023-01-01'
        self.end = datetime.now().strftime('%Y-%m-%d')
        self.filename = 'test_data.csv'

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_download_batch(self):
        """Test downloading a batch of tickers."""
        df = download_batch(self.tickers, self.start, self.end)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertTrue(all(col in df.columns for col in self.tickers))

    def test_download_and_save(self):
        """Test downloading and saving data."""
        download_and_save(self.tickers, self.start, self.end, self.filename)
        self.assertTrue(os.path.exists(self.filename))
        df = pd.read_csv(self.filename)
        self.assertFalse(df.empty)
        self.assertTrue(all(col in df.columns for col in self.tickers))

    def test_download_and_save_existing_file(self):
        """Test skipping download if file exists."""
        with open(self.filename, 'w') as f:
            f.write('test')
        download_and_save(self.tickers, self.start, self.end, self.filename)
        with open(self.filename, 'r') as f:
            self.assertEqual(f.read(), 'test')  # File unchanged

if __name__ == '__main__':
    unittest.main()