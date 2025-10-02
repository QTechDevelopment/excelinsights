# Excel Insights - Feature Documentation

## Overview

Excel Insights is a powerful tool for analyzing and comparing Excel files, with advanced capabilities for tracking changes across multiple versions over time.

## Key Features

### 1. Two-File Comparison (Standard Mode)

Compare two Excel files side-by-side with detailed difference highlighting.

**Features:**
- Upload two Excel files
- Select sheets to compare
- View differences and matches highlighted in color
- Export detailed comparison reports
- Natural language queries for analysis

**Use Cases:**
- Compare two versions of a document
- Validate data between source and target files
- Identify discrepancies in data sets

### 2. Multi-File Version Tracking (3+ Files)

Track changes across multiple versions of Excel files to understand how data evolves over time.

**Features:**
- Upload 3 or more Excel files (chronologically ordered)
- Track cell-level changes across all versions
- Visualize change timeline and trends
- Identify most frequently changed columns
- View complete change history for each cell
- Export comprehensive change reports

**Use Cases:**
- Track monthly/quarterly data updates
- Monitor inventory or pricing changes over time
- Audit data modifications across multiple versions
- Understand data evolution patterns

### 3. Comparison Methods

Three comparison methods available:

1. **Exact Comparison** (Default)
   - Compares values exactly as they appear
   - Best for: ID fields, categorical data, precise text matching

2. **Fuzzy Comparison**
   - Uses similarity matching (80% threshold)
   - Best for: Text with minor variations, typos, formatting differences

3. **Numeric Threshold Comparison**
   - Considers numeric values equal within a threshold (default: 0.001)
   - Best for: Floating-point numbers, financial data with rounding

### 4. Highlighting and Visualization

**Color Coding:**
- 🟡 Yellow: Differences/Changes
- 🟢 Green: Matches/Unchanged

**Visualization Types:**
- Summary metrics dashboard
- Side-by-side file comparison
- Heatmap of differences
- Timeline charts showing match percentages
- Bar charts of column-wise changes
- Pie charts of overall distribution

### 5. Advanced Features

**Difference Detection:**
- Fields present in one file but not the other
- Number differences with configurable thresholds
- Text changes and variations
- Missing values and null handling

**Change Tracking:**
- Cell-level history across versions
- Column-wise change frequency
- Version transition analysis
- Unchanged cell identification

**Export Options:**
- Excel reports with multiple sheets
- Difference tables
- Match tables
- Summary statistics
- Complete change history

## How to Use

### Standard Two-File Comparison

1. **Upload Files:**
   - In the sidebar, select "Two Files (Standard)" mode
   - Upload your first Excel file
   - Upload your second Excel file

2. **Select Sheets:**
   - Choose which sheet to compare from each file
   - Both sheets should have similar structure

3. **Compare:**
   - Click "📊 Compare Files" button
   - View results in multiple tabs:
     - Visual Comparison (highlighted differences)
     - Detailed Results (comparison matrix)
     - Statistics (charts and graphs)
     - Differences & Matches (detailed tables)

4. **Export:**
   - Download Excel report with all findings
   - Copy summary to clipboard

### Multi-File Version Tracking

1. **Upload Files:**
   - In the sidebar, select "Multiple Files (3+)" mode
   - Upload your Excel files in chronological order
     - Oldest version first
     - Newest version last
   - Files should have the same sheet structure

2. **Select Sheet:**
   - Choose which sheet to compare across all files
   - Must exist in all uploaded files

3. **Compare Versions:**
   - Click "🔄 Compare All Versions" button
   - View comprehensive results in tabs:
     - **Version Timeline**: Match percentage trends
     - **Pairwise Comparisons**: Consecutive version differences
     - **Change Tracking**: Cell-level history
     - **Statistics**: Multi-file analytics

4. **Export:**
   - Click "📥 Export All Data" to download:
     - All version data
     - Complete change history
     - Summary statistics

### Natural Language Queries

Use natural language to perform analyses:

**Example Queries:**
- "Compare these two files and highlight differences in yellow, matches in green"
- "Show me all rows where values are different between files"
- "Find matching values in both datasets"
- "Highlight cells that don't match between sheets"
- "Compare data and show summary statistics"

## Tips and Best Practices

### For Accurate Comparisons:

1. **Consistent Structure:**
   - Ensure files have the same column names
   - Keep data types consistent across versions

2. **Chronological Order:**
   - For multi-file tracking, upload files in time order
   - Use consistent naming (e.g., file_v1, file_v2, file_v3)

3. **Data Preparation:**
   - Remove empty rows/columns before uploading
   - Ensure column headers are in the first row

4. **Version Labels:**
   - Use descriptive file names (e.g., "products_2024_01.xlsx")
   - Include dates or version numbers in filenames

### Performance Considerations:

- **File Size:** Works best with files under 1MB
- **Number of Files:** Optimal performance with 3-10 files
- **Rows:** Handles up to 10,000 rows efficiently
- **Columns:** Works well with up to 100 columns

### Understanding Results:

**Match Percentage:**
- 100% = Files are identical
- 90-99% = Minor differences
- 70-89% = Moderate differences
- Below 70% = Significant differences

**Change Rate (Multi-File):**
- Shows % of cells that changed across versions
- Higher % indicates more volatility
- Track specific columns for targeted analysis

## Troubleshooting

### Common Issues:

**"No common columns found"**
- Ensure files have matching column names
- Check for extra spaces in column headers
- Verify sheet names are correct

**"Not enough data to compare"**
- Ensure selected sheet exists in all files
- Check that sheets contain data

**"Shape mismatch warning"**
- Files have different number of rows/columns
- Comparison will still work (using common columns)
- Review compatibility warnings

### Getting Help:

For issues or feature requests, please check the documentation or submit an issue on GitHub.

## Examples

### Example 1: Monthly Price Tracking

Track price changes across monthly product catalogs:

1. Upload: `products_jan.xlsx`, `products_feb.xlsx`, `products_mar.xlsx`
2. Select: "Products" sheet
3. Compare: View which products had price changes
4. Export: Generate report showing price evolution

### Example 2: Inventory Audit

Compare inventory counts from different systems:

1. Upload two inventory files
2. Compare with numeric threshold method
3. Identify discrepancies
4. Export difference report for investigation

### Example 3: Data Quality Validation

Track data quality improvements over time:

1. Upload historical data snapshots
2. Monitor missing value trends
3. Track data completeness
4. Generate quality improvement reports

## Advanced Configuration

### Comparison Parameters:

- **method**: 'exact', 'fuzzy', or 'numeric_threshold'
- **numeric_threshold**: Tolerance for numeric comparisons (default: 0.001)
- **highlight_differences**: Show differences in color
- **highlight_matches**: Show matches in color
- **diff_color**: Color for differences (default: yellow)
- **match_color**: Color for matches (default: green)

These can be configured through the natural language interface or directly in the code.

## Technical Details

### Supported Formats:
- `.xlsx` (Excel 2007+)
- `.xls` (Excel 97-2003)

### Dependencies:
- pandas: Data manipulation
- openpyxl: Excel file handling
- streamlit: Web interface
- plotly: Visualizations

### Architecture:
- **ExcelProcessor**: File loading and validation
- **ComparisonEngine**: Core comparison logic
- **VisualizationEngine**: Display and charts
- **NLPProcessor**: Natural language parsing

## Future Enhancements

Planned features:
- PDF export support
- Custom comparison rules
- Automated scheduling for periodic comparisons
- API for programmatic access
- Advanced filtering and search
- Conditional highlighting rules
- Collaboration features

---

**Version:** 1.0.0  
**Last Updated:** 2024
