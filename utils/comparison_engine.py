import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import difflib

class ComparisonEngine:
    """Handle data comparison operations between DataFrames"""
    
    def __init__(self):
        self.comparison_methods = ['exact', 'fuzzy', 'numeric_threshold']
    
    def compare_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                          method: str = 'exact', 
                          numeric_threshold: float = 0.001,
                          highlight_differences: bool = True,
                          highlight_matches: bool = True) -> Dict[str, Any]:
        """
        Compare two DataFrames and return detailed comparison results
        
        Args:
            df1, df2: DataFrames to compare
            method: Comparison method ('exact', 'fuzzy', 'numeric_threshold')
            numeric_threshold: Threshold for numeric comparisons
            highlight_differences: Whether to highlight differences
            highlight_matches: Whether to highlight matches
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            # Prepare DataFrames for comparison
            aligned_df1, aligned_df2 = self._align_dataframes(df1, df2)
            
            # Perform comparison based on method
            if method == 'exact':
                comparison_result = self._exact_comparison(aligned_df1, aligned_df2)
            elif method == 'fuzzy':
                comparison_result = self._fuzzy_comparison(aligned_df1, aligned_df2)
            elif method == 'numeric_threshold':
                comparison_result = self._numeric_threshold_comparison(
                    aligned_df1, aligned_df2, numeric_threshold
                )
            else:
                raise ValueError(f"Unknown comparison method: {method}")
            
            # Generate summary statistics
            summary = self._generate_comparison_summary(comparison_result, aligned_df1, aligned_df2)
            
            # Prepare highlighted DataFrames if requested
            highlighted_data = {}
            if highlight_differences or highlight_matches:
                highlighted_data = self._create_highlighted_dataframes(
                    aligned_df1, aligned_df2, comparison_result,
                    highlight_differences, highlight_matches
                )
            
            return {
                'method': method,
                'comparison_matrix': comparison_result,
                'summary': summary,
                'highlighted_data': highlighted_data,
                'differences': self._extract_differences(aligned_df1, aligned_df2, comparison_result),
                'matches': self._extract_matches(aligned_df1, aligned_df2, comparison_result),
                'df1_aligned': aligned_df1,
                'df2_aligned': aligned_df2
            }
            
        except Exception as e:
            raise Exception(f"Comparison failed: {str(e)}")
    
    def _align_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align two DataFrames for comparison by ensuring same shape and columns
        
        Args:
            df1, df2: DataFrames to align
            
        Returns:
            Tuple of aligned DataFrames
        """
        # Find common columns
        common_columns = list(set(df1.columns) & set(df2.columns))
        
        if not common_columns:
            raise ValueError("No common columns found between DataFrames")
        
        # Select only common columns
        aligned_df1 = df1[common_columns].copy()
        aligned_df2 = df2[common_columns].copy()
        
        # Ensure same number of rows by padding with NaN
        max_rows = max(len(aligned_df1), len(aligned_df2))
        
        if len(aligned_df1) < max_rows:
            padding_rows = max_rows - len(aligned_df1)
            padding_df = pd.DataFrame(np.nan, index=range(padding_rows), columns=aligned_df1.columns)
            aligned_df1 = pd.concat([aligned_df1, padding_df], ignore_index=True)
        
        if len(aligned_df2) < max_rows:
            padding_rows = max_rows - len(aligned_df2)
            padding_df = pd.DataFrame(np.nan, index=range(padding_rows), columns=aligned_df2.columns)
            aligned_df2 = pd.concat([aligned_df2, padding_df], ignore_index=True)
        
        # Reorder columns to match
        aligned_df1 = aligned_df1.reindex(columns=common_columns)
        aligned_df2 = aligned_df2.reindex(columns=common_columns)
        
        return aligned_df1, aligned_df2
    
    def _exact_comparison(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Perform exact comparison between DataFrames
        
        Args:
            df1, df2: Aligned DataFrames to compare
            
        Returns:
            Boolean DataFrame indicating matches (True) and differences (False)
        """
        # Handle NaN values - consider NaN == NaN as True
        comparison_result = pd.DataFrame(index=df1.index, columns=df1.columns, dtype=bool)
        
        for col in df1.columns:
            col1_values = df1[col]
            col2_values = df2[col]
            
            # Handle different data types
            if col1_values.dtype != col2_values.dtype:
                # Convert to string for comparison if types differ
                col1_values = col1_values.astype(str)
                col2_values = col2_values.astype(str)
            
            # Compare values, treating NaN as equal
            mask_both_nan = col1_values.isna() & col2_values.isna()
            mask_equal = col1_values == col2_values
            
            comparison_result[col] = mask_both_nan | mask_equal
        
        return comparison_result
    
    def _fuzzy_comparison(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                         similarity_threshold: float = 0.8) -> pd.DataFrame:
        """
        Perform fuzzy string comparison between DataFrames
        
        Args:
            df1, df2: Aligned DataFrames to compare
            similarity_threshold: Minimum similarity ratio for match
            
        Returns:
            Boolean DataFrame indicating fuzzy matches
        """
        comparison_result = pd.DataFrame(index=df1.index, columns=df1.columns, dtype=bool)
        
        for col in df1.columns:
            col1_values = df1[col].astype(str)
            col2_values = df2[col].astype(str)
            
            matches = []
            for val1, val2 in zip(col1_values, col2_values):
                if pd.isna(val1) and pd.isna(val2):
                    matches.append(True)
                elif pd.isna(val1) or pd.isna(val2):
                    matches.append(False)
                else:
                    similarity = difflib.SequenceMatcher(None, str(val1), str(val2)).ratio()
                    matches.append(similarity >= similarity_threshold)
            
            comparison_result[col] = matches
        
        return comparison_result
    
    def _numeric_threshold_comparison(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                    threshold: float = 0.001) -> pd.DataFrame:
        """
        Perform numeric comparison with threshold tolerance
        
        Args:
            df1, df2: Aligned DataFrames to compare
            threshold: Numeric threshold for considering values equal
            
        Returns:
            Boolean DataFrame indicating matches within threshold
        """
        comparison_result = pd.DataFrame(index=df1.index, columns=df1.columns, dtype=bool)
        
        for col in df1.columns:
            col1_values = df1[col]
            col2_values = df2[col]
            
            # Try to convert to numeric, fallback to exact comparison for non-numeric
            try:
                col1_numeric = pd.to_numeric(col1_values, errors='coerce')
                col2_numeric = pd.to_numeric(col2_values, errors='coerce')
                
                # Handle NaN values
                mask_both_nan = col1_numeric.isna() & col2_numeric.isna()
                mask_both_valid = col1_numeric.notna() & col2_numeric.notna()
                
                # Calculate differences for valid numeric values
                differences = np.abs(col1_numeric - col2_numeric)
                mask_within_threshold = differences <= threshold
                
                # Combine conditions
                comparison_result[col] = mask_both_nan | (mask_both_valid & mask_within_threshold)
                
            except:
                # Fallback to exact comparison for non-numeric data
                comparison_result[col] = self._exact_comparison(
                    df1[[col]], df2[[col]]
                )[col]
        
        return comparison_result
    
    def _generate_comparison_summary(self, comparison_result: pd.DataFrame, 
                                   df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for comparison
        
        Args:
            comparison_result: Boolean DataFrame with comparison results
            df1, df2: Original DataFrames
            
        Returns:
            Dictionary with summary statistics
        """
        total_cells = comparison_result.size
        matching_cells = comparison_result.sum().sum()
        different_cells = total_cells - matching_cells
        
        # Calculate per-column statistics
        column_stats = {}
        for col in comparison_result.columns:
            col_matches = comparison_result[col].sum()
            col_total = len(comparison_result[col])
            column_stats[col] = {
                'matches': int(col_matches),
                'differences': int(col_total - col_matches),
                'match_percentage': float(col_matches / col_total * 100) if col_total > 0 else 0
            }
        
        return {
            'total_cells': int(total_cells),
            'matching_cells': int(matching_cells),
            'different_cells': int(different_cells),
            'match_percentage': float(matching_cells / total_cells * 100) if total_cells > 0 else 0,
            'column_statistics': column_stats,
            'df1_shape': df1.shape,
            'df2_shape': df2.shape,
            'common_columns': list(comparison_result.columns)
        }
    
    def _create_highlighted_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                     comparison_result: pd.DataFrame,
                                     highlight_differences: bool,
                                     highlight_matches: bool) -> Dict[str, pd.DataFrame]:
        """
        Create DataFrames with highlighting information for visualization
        
        Args:
            df1, df2: Original DataFrames
            comparison_result: Boolean comparison matrix
            highlight_differences: Whether to mark differences
            highlight_matches: Whether to mark matches
            
        Returns:
            Dictionary with highlighted DataFrames
        """
        highlighted_data = {}
        
        # Create highlighting matrices
        df1_highlight = pd.DataFrame('', index=df1.index, columns=df1.columns)
        df2_highlight = pd.DataFrame('', index=df2.index, columns=df2.columns)
        
        for col in comparison_result.columns:
            for idx in comparison_result.index:
                is_match = comparison_result.loc[idx, col]
                
                if is_match and highlight_matches:
                    df1_highlight.loc[idx, col] = 'match'
                    df2_highlight.loc[idx, col] = 'match'
                elif not is_match and highlight_differences:
                    df1_highlight.loc[idx, col] = 'difference'
                    df2_highlight.loc[idx, col] = 'difference'
        
        highlighted_data['df1_data'] = df1
        highlighted_data['df2_data'] = df2
        highlighted_data['df1_highlight'] = df1_highlight
        highlighted_data['df2_highlight'] = df2_highlight
        
        return highlighted_data
    
    def _extract_differences(self, df1: pd.DataFrame, df2: pd.DataFrame,
                           comparison_result: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rows/cells that are different between DataFrames
        
        Args:
            df1, df2: Original DataFrames
            comparison_result: Boolean comparison matrix
            
        Returns:
            DataFrame with difference information
        """
        differences = []
        
        for col in comparison_result.columns:
            for idx in comparison_result.index:
                if not comparison_result.loc[idx, col]:
                    differences.append({
                        'row': idx,
                        'column': col,
                        'file1_value': df1.loc[idx, col] if idx < len(df1) else None,
                        'file2_value': df2.loc[idx, col] if idx < len(df2) else None
                    })
        
        return pd.DataFrame(differences) if differences else pd.DataFrame()
    
    def _extract_matches(self, df1: pd.DataFrame, df2: pd.DataFrame,
                        comparison_result: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rows/cells that match between DataFrames
        
        Args:
            df1, df2: Original DataFrames
            comparison_result: Boolean comparison matrix
            
        Returns:
            DataFrame with match information
        """
        matches = []
        
        for col in comparison_result.columns:
            for idx in comparison_result.index:
                if comparison_result.loc[idx, col]:
                    matches.append({
                        'row': idx,
                        'column': col,
                        'value': df1.loc[idx, col] if idx < len(df1) else None
                    })
        
        return pd.DataFrame(matches) if matches else pd.DataFrame()
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform single DataFrame analysis
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'summary_stats': {},
            'duplicate_rows': df.duplicated().sum(),
            'unique_values_per_column': {}
        }
        
        # Generate summary statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            analysis['summary_stats'] = df[numeric_columns].describe().to_dict()
        
        # Count unique values per column
        for col in df.columns:
            analysis['unique_values_per_column'][col] = df[col].nunique()
        
        return analysis
    
    def compare_multiple_dataframes(self, dataframes: List[pd.DataFrame], 
                                   labels: Optional[List[str]] = None,
                                   method: str = 'exact',
                                   numeric_threshold: float = 0.001) -> Dict[str, Any]:
        """
        Compare multiple DataFrames (3 or more) and track changes across versions
        
        Args:
            dataframes: List of DataFrames to compare
            labels: Optional labels for each DataFrame (e.g., timestamps, versions)
            method: Comparison method ('exact', 'fuzzy', 'numeric_threshold')
            numeric_threshold: Threshold for numeric comparisons
            
        Returns:
            Dictionary containing multi-file comparison results
        """
        if len(dataframes) < 2:
            raise ValueError("At least 2 DataFrames are required for comparison")
        
        # Generate default labels if not provided
        if labels is None:
            labels = [f"File {i+1}" for i in range(len(dataframes))]
        elif len(labels) != len(dataframes):
            raise ValueError("Number of labels must match number of DataFrames")
        
        try:
            # Find common columns across all DataFrames
            common_columns = set(dataframes[0].columns)
            for df in dataframes[1:]:
                common_columns &= set(df.columns)
            common_columns = list(common_columns)
            
            if not common_columns:
                raise ValueError("No common columns found across all DataFrames")
            
            # Align all DataFrames
            aligned_dfs = []
            max_rows = max(len(df) for df in dataframes)
            
            for df in dataframes:
                aligned_df = df[common_columns].copy()
                if len(aligned_df) < max_rows:
                    padding_rows = max_rows - len(aligned_df)
                    padding_df = pd.DataFrame(np.nan, index=range(padding_rows), columns=aligned_df.columns)
                    aligned_df = pd.concat([aligned_df, padding_df], ignore_index=True)
                aligned_dfs.append(aligned_df)
            
            # Perform pairwise comparisons
            pairwise_comparisons = {}
            for i in range(len(aligned_dfs) - 1):
                comparison_key = f"{labels[i]}_vs_{labels[i+1]}"
                pairwise_comparisons[comparison_key] = self.compare_dataframes(
                    aligned_dfs[i], aligned_dfs[i+1], 
                    method=method,
                    numeric_threshold=numeric_threshold,
                    highlight_differences=True,
                    highlight_matches=False
                )
            
            # Track changes across all versions
            change_tracking = self._track_changes_across_versions(
                aligned_dfs, labels, common_columns, method, numeric_threshold
            )
            
            # Generate comprehensive summary
            multi_summary = self._generate_multi_file_summary(
                aligned_dfs, labels, pairwise_comparisons, change_tracking
            )
            
            return {
                'method': method,
                'labels': labels,
                'common_columns': common_columns,
                'pairwise_comparisons': pairwise_comparisons,
                'change_tracking': change_tracking,
                'summary': multi_summary,
                'aligned_dataframes': {label: df for label, df in zip(labels, aligned_dfs)}
            }
            
        except Exception as e:
            raise Exception(f"Multi-file comparison failed: {str(e)}")
    
    def _track_changes_across_versions(self, aligned_dfs: List[pd.DataFrame], 
                                      labels: List[str],
                                      common_columns: List[str],
                                      method: str,
                                      numeric_threshold: float) -> Dict[str, Any]:
        """
        Track how each cell changes across all versions
        
        Args:
            aligned_dfs: List of aligned DataFrames
            labels: Labels for each DataFrame
            common_columns: List of common columns
            method: Comparison method
            numeric_threshold: Numeric threshold
            
        Returns:
            Dictionary with change tracking information
        """
        change_tracking = {
            'cell_history': {},  # History of each cell across versions
            'changed_cells': [],  # Cells that changed at least once
            'unchanged_cells': [],  # Cells that never changed
            'column_change_frequency': {}  # How often each column changes
        }
        
        # Track changes for each cell position
        for col in common_columns:
            column_changes = 0
            
            for row_idx in range(len(aligned_dfs[0])):
                cell_key = f"row_{row_idx}_col_{col}"
                cell_values = []
                changed = False
                
                # Collect values across all versions
                for i, df in enumerate(aligned_dfs):
                    value = df.loc[row_idx, col]
                    cell_values.append({
                        'version': labels[i],
                        'value': value
                    })
                    
                    # Check if value changed from previous version
                    if i > 0:
                        prev_value = aligned_dfs[i-1].loc[row_idx, col]
                        if not self._values_equal(value, prev_value, method, numeric_threshold):
                            changed = True
                
                change_tracking['cell_history'][cell_key] = {
                    'row': row_idx,
                    'column': col,
                    'values': cell_values,
                    'changed': changed
                }
                
                if changed:
                    change_tracking['changed_cells'].append({
                        'row': row_idx,
                        'column': col,
                        'values': cell_values
                    })
                    column_changes += 1
                else:
                    change_tracking['unchanged_cells'].append({
                        'row': row_idx,
                        'column': col,
                        'value': cell_values[0]['value'] if cell_values else None
                    })
            
            change_tracking['column_change_frequency'][col] = column_changes
        
        return change_tracking
    
    def _values_equal(self, val1: Any, val2: Any, method: str, threshold: float) -> bool:
        """
        Check if two values are equal based on comparison method
        
        Args:
            val1, val2: Values to compare
            method: Comparison method
            threshold: Numeric threshold
            
        Returns:
            True if values are equal, False otherwise
        """
        # Handle NaN values
        if pd.isna(val1) and pd.isna(val2):
            return True
        if pd.isna(val1) or pd.isna(val2):
            return False
        
        if method == 'numeric_threshold':
            # Try numeric comparison
            try:
                num1 = pd.to_numeric(val1, errors='coerce')
                num2 = pd.to_numeric(val2, errors='coerce')
                if pd.notna(num1) and pd.notna(num2):
                    return abs(num1 - num2) <= threshold
            except:
                pass
        elif method == 'fuzzy':
            # Fuzzy string comparison
            similarity = difflib.SequenceMatcher(None, str(val1), str(val2)).ratio()
            return similarity >= 0.8
        
        # Default exact comparison
        return val1 == val2
    
    def _generate_multi_file_summary(self, aligned_dfs: List[pd.DataFrame],
                                    labels: List[str],
                                    pairwise_comparisons: Dict[str, Any],
                                    change_tracking: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for multi-file comparison
        
        Args:
            aligned_dfs: List of aligned DataFrames
            labels: Labels for each DataFrame
            pairwise_comparisons: Pairwise comparison results
            change_tracking: Change tracking information
            
        Returns:
            Dictionary with summary statistics
        """
        total_cells = aligned_dfs[0].size
        changed_cells = len(change_tracking['changed_cells'])
        unchanged_cells = len(change_tracking['unchanged_cells'])
        
        # Calculate change statistics per version transition
        version_transitions = []
        for comparison_key, result in pairwise_comparisons.items():
            version_transitions.append({
                'transition': comparison_key,
                'differences': result['summary']['different_cells'],
                'matches': result['summary']['matching_cells'],
                'match_percentage': result['summary']['match_percentage']
            })
        
        return {
            'num_versions': len(aligned_dfs),
            'version_labels': labels,
            'total_cells_per_version': total_cells,
            'total_changed_cells': changed_cells,
            'total_unchanged_cells': unchanged_cells,
            'change_percentage': (changed_cells / total_cells * 100) if total_cells > 0 else 0,
            'version_transitions': version_transitions,
            'most_changed_columns': sorted(
                change_tracking['column_change_frequency'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'common_columns': list(aligned_dfs[0].columns),
            'shapes': {label: df.shape for label, df in zip(labels, aligned_dfs)}
        }
