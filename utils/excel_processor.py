import pandas as pd
import streamlit as st
from typing import Dict, Any
import io

class ExcelProcessor:
    """Handle Excel file loading and processing operations"""
    
    def __init__(self):
        self.supported_formats = ['.xlsx', '.xls']
    
    def load_excel(self, uploaded_file) -> Dict[str, pd.DataFrame]:
        """
        Load Excel file and return dictionary of DataFrames for each sheet
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dictionary with sheet names as keys and DataFrames as values
        """
        try:
            # Read all sheets from the Excel file
            excel_data = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')
            
            # Clean and process each sheet
            processed_data = {}
            for sheet_name, df in excel_data.items():
                # Basic cleaning
                processed_df = self._clean_dataframe(df)
                processed_data[sheet_name] = processed_df
            
            return processed_data
            
        except Exception as e:
            raise Exception(f"Failed to load Excel file: {str(e)}")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare DataFrame for analysis
        
        Args:
            df: Raw DataFrame from Excel
            
        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Remove completely empty rows and columns
        cleaned_df = cleaned_df.dropna(how='all').dropna(axis=1, how='all')
        
        # Reset index
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        # Ensure column names are strings
        cleaned_df.columns = cleaned_df.columns.astype(str)
        
        # Remove any leading/trailing whitespace from string columns
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                # Convert back to original type if it was mistakenly converted
                cleaned_df[col] = cleaned_df[col].replace('nan', pd.NA)
        
        return cleaned_df
    
    def get_sheet_info(self, excel_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Get information about each sheet in the Excel file
        
        Args:
            excel_data: Dictionary of DataFrames
            
        Returns:
            Dictionary with sheet information
        """
        sheet_info = {}
        
        for sheet_name, df in excel_data.items():
            sheet_info[sheet_name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
            }
        
        return sheet_info
    
    def validate_comparison_compatibility(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if two DataFrames are compatible for comparison
        
        Args:
            df1, df2: DataFrames to compare
            
        Returns:
            Dictionary with compatibility information
        """
        compatibility = {
            'compatible': True,
            'warnings': [],
            'shape_match': df1.shape == df2.shape,
            'column_match': list(df1.columns) == list(df2.columns),
            'common_columns': list(set(df1.columns) & set(df2.columns)),
            'df1_only_columns': list(set(df1.columns) - set(df2.columns)),
            'df2_only_columns': list(set(df2.columns) - set(df1.columns))
        }
        
        # Add warnings based on compatibility issues
        if not compatibility['shape_match']:
            compatibility['warnings'].append(f"Shape mismatch: {df1.shape} vs {df2.shape}")
        
        if not compatibility['column_match']:
            compatibility['warnings'].append("Column names don't match exactly")
        
        if len(compatibility['common_columns']) == 0:
            compatibility['compatible'] = False
            compatibility['warnings'].append("No common columns found")
        
        return compatibility
    
    def export_to_excel(self, data: Dict[str, pd.DataFrame], filename: str = "exported_data.xlsx") -> io.BytesIO:
        """
        Export dictionary of DataFrames to Excel file
        
        Args:
            data: Dictionary with DataFrames to export
            filename: Name for the output file
            
        Returns:
            BytesIO object containing the Excel file
        """
        output = io.BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, df in data.items():
                    # Ensure sheet name is valid for Excel
                    safe_sheet_name = self._sanitize_sheet_name(sheet_name)
                    df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            
            output.seek(0)
            return output
            
        except Exception as e:
            raise Exception(f"Failed to export to Excel: {str(e)}")
    
    def _sanitize_sheet_name(self, name: str) -> str:
        """
        Sanitize sheet name to be valid for Excel
        
        Args:
            name: Original sheet name
            
        Returns:
            Sanitized sheet name
        """
        # Remove invalid characters for Excel sheet names
        invalid_chars = ['\\', '/', '*', '?', ':', '[', ']']
        sanitized = name
        
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Limit length to 31 characters (Excel limit)
        if len(sanitized) > 31:
            sanitized = sanitized[:31]
        
        return sanitized
