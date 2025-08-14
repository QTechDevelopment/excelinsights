# Overview

Excel File Analyzer & Comparator is a Streamlit-based web application that provides advanced Excel file analysis, comparison, and querying capabilities. The tool allows users to upload Excel files, perform detailed comparisons between datasets, analyze data using natural language commands, and visualize results through interactive charts and tables. The application is designed to handle complex data comparison scenarios with various comparison methods including exact matching, fuzzy matching, and numeric threshold comparisons.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and deployment
- **Layout**: Wide layout configuration with sidebar navigation for file uploads
- **State Management**: Streamlit session state for persistent data storage across user interactions
- **User Interface**: Tab-based interface for organizing different views (Visual Comparison, Detailed Results, Statistics, Differences & Matches)

## Backend Architecture
- **Modular Design**: Utility-based architecture with specialized processors for different functionalities
- **Core Components**:
  - `ExcelProcessor`: Handles file loading, validation, and data cleaning operations
  - `ComparisonEngine`: Manages data comparison logic with multiple comparison methods
  - `NLPProcessor`: Processes natural language queries using pattern matching and command interpretation
  - `VisualizationEngine`: Generates interactive charts and styled data displays

## Data Processing Pipeline
- **File Processing**: Excel files are loaded using pandas with openpyxl engine for comprehensive Excel support
- **Data Alignment**: DataFrames are aligned for comparison with handling of structural differences
- **Comparison Methods**: Three primary comparison approaches - exact matching, fuzzy matching, and numeric threshold-based comparison
- **Result Generation**: Comprehensive comparison results with summary statistics and highlighted differences

## Natural Language Processing
- **Pattern Matching**: Regular expression-based command recognition for common analysis tasks
- **Command Categories**: Support for compare, analyze, filter, and search operations
- **Color and Highlight Mapping**: Semantic understanding of color preferences and highlight types for result visualization

## Visualization System
- **Charting Library**: Plotly integration for interactive data visualizations
- **Color Scheme**: Predefined color mapping for consistent visual representation
- **Multi-view Display**: Tabbed interface presenting different perspectives of comparison results
- **Styled DataFrames**: Custom styling for highlighting differences and matches in tabular data

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis library for Excel processing
- **Plotly**: Interactive visualization library for charts and graphs
- **NumPy**: Numerical computing library for mathematical operations

## File Processing
- **OpenPyXL**: Excel file reading and writing engine
- **io**: Built-in Python library for handling file streams
- **base64**: Built-in Python library for encoding operations

## Text Processing
- **difflib**: Built-in Python library for sequence comparison and fuzzy matching
- **re**: Built-in Python regular expressions library for pattern matching

## Development Tools
- **typing**: Built-in Python library for type hints and annotations

The application is designed to be self-contained with minimal external service dependencies, focusing on local file processing and analysis capabilities.