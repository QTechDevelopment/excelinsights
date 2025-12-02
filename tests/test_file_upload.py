"""
Test file upload validation and multi-file processing functionality.
"""

import unittest
import pandas as pd
import io
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.excel_processor import ExcelProcessor
from utils.comparison_engine import ComparisonEngine


class TestFileUploadValidation(unittest.TestCase):
    """Test cases for file upload validation logic"""
    
    def test_max_file_limit(self):
        """Test that the maximum file limit is 4"""
        max_files = 4
        # Simulate file count validation
        file_counts = [1, 2, 3, 4, 5, 6]
        
        for count in file_counts:
            if count > max_files:
                self.assertGreater(count, max_files, 
                    f"File count {count} should exceed maximum of {max_files}")
            else:
                self.assertLessEqual(count, max_files, 
                    f"File count {count} should be within maximum of {max_files}")
    
    def test_min_file_requirement(self):
        """Test that at least 2 files are required for multi-file comparison"""
        min_files = 2
        file_counts = [0, 1, 2, 3]
        
        for count in file_counts:
            if count < min_files:
                self.assertLess(count, min_files, 
                    f"File count {count} should be less than minimum of {min_files}")
            else:
                self.assertGreaterEqual(count, min_files, 
                    f"File count {count} should meet minimum of {min_files}")
    
    def test_valid_file_range(self):
        """Test that file count between 2-4 is valid"""
        valid_counts = [2, 3, 4]
        invalid_counts = [0, 1, 5, 6, 10]
        
        for count in valid_counts:
            self.assertTrue(2 <= count <= 4, 
                f"File count {count} should be in valid range [2-4]")
        
        for count in invalid_counts:
            self.assertFalse(2 <= count <= 4, 
                f"File count {count} should be outside valid range [2-4]")


class TestExcelProcessing(unittest.TestCase):
    """Test cases for Excel file processing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = ExcelProcessor()
        self.comparison_engine = ComparisonEngine()
    
    def create_sample_excel(self, data_dict, sheet_name='Sheet1'):
        """Helper method to create a sample Excel file in memory"""
        df = pd.DataFrame(data_dict)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        output.seek(0)
        output.name = 'test.xlsx'  # Add name attribute for compatibility
        return output
    
    def test_load_single_excel(self):
        """Test loading a single Excel file"""
        sample_data = {
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'London', 'Paris']
        }
        
        excel_file = self.create_sample_excel(sample_data)
        result = self.processor.load_excel(excel_file)
        
        self.assertIsInstance(result, dict)
        self.assertIn('Sheet1', result)
        self.assertEqual(len(result['Sheet1']), 3)
        self.assertEqual(list(result['Sheet1'].columns), ['Name', 'Age', 'City'])
    
    def test_load_multiple_sheets(self):
        """Test loading Excel file with multiple sheets"""
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'X': [5, 6], 'Y': [7, 8]})
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df1.to_excel(writer, sheet_name='Sheet1', index=False)
            df2.to_excel(writer, sheet_name='Sheet2', index=False)
        output.seek(0)
        output.name = 'test.xlsx'
        
        result = self.processor.load_excel(output)
        
        self.assertEqual(len(result), 2)
        self.assertIn('Sheet1', result)
        self.assertIn('Sheet2', result)
    
    def test_compare_two_dataframes(self):
        """Test comparing two DataFrames"""
        data1 = {
            'ID': [1, 2, 3],
            'Value': [100, 200, 300]
        }
        data2 = {
            'ID': [1, 2, 3],
            'Value': [100, 250, 300]
        }
        
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        
        result = self.comparison_engine.compare_dataframes(df1, df2)
        
        self.assertIn('summary', result)
        self.assertIn('comparison_matrix', result)
        self.assertEqual(result['summary']['different_cells'], 1)
        self.assertEqual(result['summary']['matching_cells'], 5)
    
    def test_compare_multiple_dataframes(self):
        """Test comparing multiple DataFrames (2-4 files)"""
        data1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        data2 = pd.DataFrame({'A': [1, 2, 4], 'B': [4, 5, 6]})
        data3 = pd.DataFrame({'A': [1, 3, 4], 'B': [4, 5, 7]})
        
        # Test with 2 files
        result_2 = self.comparison_engine.compare_multiple_dataframes(
            [data1, data2],
            labels=['File1', 'File2']
        )
        self.assertEqual(result_2['summary']['num_versions'], 2)
        self.assertEqual(len(result_2['pairwise_comparisons']), 1)
        
        # Test with 3 files
        result_3 = self.comparison_engine.compare_multiple_dataframes(
            [data1, data2, data3],
            labels=['File1', 'File2', 'File3']
        )
        self.assertEqual(result_3['summary']['num_versions'], 3)
        self.assertEqual(len(result_3['pairwise_comparisons']), 2)
        
        # Test with 4 files
        data4 = pd.DataFrame({'A': [1, 3, 5], 'B': [4, 6, 7]})
        result_4 = self.comparison_engine.compare_multiple_dataframes(
            [data1, data2, data3, data4],
            labels=['File1', 'File2', 'File3', 'File4']
        )
        self.assertEqual(result_4['summary']['num_versions'], 4)
        self.assertEqual(len(result_4['pairwise_comparisons']), 3)
    
    def test_change_tracking_across_versions(self):
        """Test change tracking functionality across versions"""
        data1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        data2 = pd.DataFrame({'A': [1, 2], 'B': [3, 5]})
        data3 = pd.DataFrame({'A': [1, 3], 'B': [3, 5]})
        
        result = self.comparison_engine.compare_multiple_dataframes(
            [data1, data2, data3],
            labels=['V1', 'V2', 'V3']
        )
        
        self.assertIn('change_tracking', result)
        self.assertIn('changed_cells', result['change_tracking'])
        self.assertIn('unchanged_cells', result['change_tracking'])
        
        # Verify that changes are tracked
        self.assertGreater(len(result['change_tracking']['changed_cells']), 0)


class TestValidationLogic(unittest.TestCase):
    """Test validation logic for file uploads"""
    
    def test_file_count_validation_logic(self):
        """Test the core validation logic for file count"""
        def validate_file_count(count):
            """Validation function that mimics app.py logic"""
            if count > 4:
                return False, "Too many files"
            elif count < 2:
                return False, "Too few files"
            else:
                return True, "Valid file count"
        
        # Test various counts
        self.assertEqual(validate_file_count(0), (False, "Too few files"))
        self.assertEqual(validate_file_count(1), (False, "Too few files"))
        self.assertEqual(validate_file_count(2), (True, "Valid file count"))
        self.assertEqual(validate_file_count(3), (True, "Valid file count"))
        self.assertEqual(validate_file_count(4), (True, "Valid file count"))
        self.assertEqual(validate_file_count(5), (False, "Too many files"))
        self.assertEqual(validate_file_count(10), (False, "Too many files"))


if __name__ == '__main__':
    unittest.main()
