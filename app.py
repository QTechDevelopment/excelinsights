import streamlit as st
import pandas as pd
import io
import base64
from utils.excel_processor import ExcelProcessor
from utils.comparison_engine import ComparisonEngine
from utils.nlp_processor import NLPProcessor
from utils.visualization import VisualizationEngine

# Initialize session state
if 'file1_data' not in st.session_state:
    st.session_state.file1_data = None
if 'file2_data' not in st.session_state:
    st.session_state.file2_data = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'selected_sheets' not in st.session_state:
    st.session_state.selected_sheets = {}
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'multi_file_data' not in st.session_state:
    st.session_state.multi_file_data = []
if 'multi_comparison_results' not in st.session_state:
    st.session_state.multi_comparison_results = None

def main():
    st.set_page_config(
        page_title="Excel File Analyzer & Comparator",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Excel File Analyzer & Comparator")
    st.markdown("Upload Excel files to analyze, compare, and query data using natural language commands.")
    
    # Initialize processors
    excel_processor = ExcelProcessor()
    comparison_engine = ComparisonEngine()
    nlp_processor = NLPProcessor()
    viz_engine = VisualizationEngine()
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 File Upload")
        
        # Mode selection
        upload_mode = st.radio(
            "Upload Mode:",
            ["Two Files (Standard)", "Multiple Files (3+)"],
            help="Choose whether to compare 2 files or track changes across multiple versions"
        )
        
        if upload_mode == "Two Files (Standard)":
            # Original two-file upload
            uploaded_file1 = st.file_uploader(
                "Upload First Excel File",
                type=['xlsx', 'xls'],
                key="file1"
            )
            
            uploaded_file2 = st.file_uploader(
                "Upload Second Excel File (Optional for comparison)",
                type=['xlsx', 'xls'],
                key="file2"
            )
            
            # Process uploaded files
            if uploaded_file1:
                try:
                    st.session_state.file1_data = excel_processor.load_excel(uploaded_file1)
                    st.success(f"✅ Loaded: {uploaded_file1.name}")
                except Exception as e:
                    st.error(f"❌ Error loading file 1: {str(e)}")
            
            if uploaded_file2:
                try:
                    st.session_state.file2_data = excel_processor.load_excel(uploaded_file2)
                    st.success(f"✅ Loaded: {uploaded_file2.name}")
                except Exception as e:
                    st.error(f"❌ Error loading file 2: {str(e)}")
        
        else:
            # Multi-file upload
            st.info("📝 Upload files in chronological order (oldest to newest) to track changes over time")
            
            uploaded_files = st.file_uploader(
                "Upload Excel Files (3 or more)",
                type=['xlsx', 'xls'],
                accept_multiple_files=True,
                key="multi_files"
            )
            
            if uploaded_files and len(uploaded_files) >= 2:
                try:
                    st.session_state.multi_file_data = []
                    st.session_state.uploaded_files = []
                    
                    for uploaded_file in uploaded_files:
                        file_data = excel_processor.load_excel(uploaded_file)
                        st.session_state.multi_file_data.append(file_data)
                        st.session_state.uploaded_files.append(uploaded_file.name)
                        st.success(f"✅ Loaded: {uploaded_file.name}")
                    
                    st.info(f"📊 Total files loaded: {len(uploaded_files)}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading files: {str(e)}")
            elif uploaded_files:
                st.warning("⚠️ Please upload at least 2 files for comparison")
    
    # Main content area
    if st.session_state.file1_data:
        # Sheet selection for two-file mode
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("File 1 Sheet Selection")
            sheet1_options = list(st.session_state.file1_data.keys())
            selected_sheet1 = st.selectbox(
                "Select sheet from File 1:",
                sheet1_options,
                key="sheet1_select"
            )
            st.session_state.selected_sheets['sheet1'] = selected_sheet1
        
        with col2:
            if st.session_state.file2_data:
                st.subheader("File 2 Sheet Selection")
                sheet2_options = list(st.session_state.file2_data.keys())
                selected_sheet2 = st.selectbox(
                    "Select sheet from File 2:",
                    sheet2_options,
                    key="sheet2_select"
                )
                st.session_state.selected_sheets['sheet2'] = selected_sheet2
        
        # Natural Language Query Section
        st.header("🗣️ Natural Language Query")
        query_input = st.text_input(
            "Enter your query (e.g., 'compare these two compounds and highlight differences in yellow, matches in green'):",
            placeholder="Type your analysis command here..."
        )
        
        col_query1, col_query2, col_query3 = st.columns([1, 1, 2])
        
        with col_query1:
            if st.button("🔍 Execute Query", type="primary"):
                if query_input:
                    execute_query(query_input, nlp_processor, comparison_engine, viz_engine)
                else:
                    st.warning("Please enter a query first.")
        
        with col_query2:
            if st.button("📊 Compare Files"):
                if st.session_state.file2_data:
                    perform_comparison(comparison_engine, viz_engine)
                else:
                    st.warning("Please upload a second file for comparison.")
        
        with col_query3:
            st.caption("Use buttons to analyze your data")
        
        # Display data section
        st.header("📋 Data View")
        
        if st.session_state.file2_data and 'sheet2' in st.session_state.selected_sheets:
            # Side-by-side comparison view
            display_side_by_side_comparison()
        else:
            # Single file view
            display_single_file_data()
        
        # Export section
        if st.session_state.comparison_results:
            st.header("💾 Export Results")
            export_comparison_results()
    
    elif st.session_state.multi_file_data and len(st.session_state.multi_file_data) >= 2:
        # Multi-file comparison mode
        st.header("📊 Multi-File Version Tracking")
        
        # Sheet selection for multi-file mode
        st.subheader("Sheet Selection")
        
        # Get common sheets across all files
        common_sheets = set(st.session_state.multi_file_data[0].keys())
        for file_data in st.session_state.multi_file_data[1:]:
            common_sheets &= set(file_data.keys())
        
        if common_sheets:
            selected_sheet = st.selectbox(
                "Select sheet to compare across all files:",
                list(common_sheets),
                key="multi_sheet_select"
            )
            
            # Compare button
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🔄 Compare All Versions", type="primary"):
                    perform_multi_file_comparison(comparison_engine, viz_engine, selected_sheet)
            
            with col2:
                if st.button("📥 Export All Data"):
                    export_multi_file_data(selected_sheet)
            
            # Display multi-file comparison results
            if st.session_state.multi_comparison_results:
                viz_engine.display_multi_file_comparison(st.session_state.multi_comparison_results)
            
            # Display data preview
            st.header("📋 Data Preview")
            display_multi_file_preview(selected_sheet)
            
        else:
            st.error("❌ No common sheets found across all uploaded files. Please ensure all files have at least one sheet with the same name.")
    
    else:
        # Welcome screen
        st.info("👆 Please upload at least one Excel file to get started.")
        
        # Example queries section
        st.header("💡 Example Queries")
        st.markdown("""
        Here are some example natural language queries you can use:
        
        - **"Compare these two compounds and highlight differences in yellow, matches in green"**
        - **"Show me all rows where column A is different between files"**
        - **"Find matching values in both datasets"**
        - **"Highlight cells that don't match between sheets"**
        - **"Compare data and show summary statistics"**
        """)

def execute_query(query, nlp_processor, comparison_engine, viz_engine):
    """Execute natural language query"""
    try:
        with st.spinner("Processing your query..."):
            # Parse the query
            parsed_query = nlp_processor.parse_query(query)
            
            # Get data based on selection
            if 'sheet1' in st.session_state.selected_sheets:
                data1 = st.session_state.file1_data[st.session_state.selected_sheets['sheet1']]
            else:
                data1 = list(st.session_state.file1_data.values())[0]
            
            data2 = None
            if st.session_state.file2_data and 'sheet2' in st.session_state.selected_sheets:
                data2 = st.session_state.file2_data[st.session_state.selected_sheets['sheet2']]
            
            # Execute based on query type
            if parsed_query['action'] == 'compare' and data2 is not None:
                results = comparison_engine.compare_dataframes(
                    data1, data2, 
                    highlight_differences=parsed_query.get('highlight_differences', True),
                    highlight_matches=parsed_query.get('highlight_matches', True)
                )
                st.session_state.comparison_results = results
                
                # Display results
                viz_engine.display_comparison_results(results, parsed_query)
                
            elif parsed_query['action'] == 'analyze':
                # Single file analysis
                results = comparison_engine.analyze_dataframe(data1)
                viz_engine.display_analysis_results(results)
                
            else:
                st.warning("Query not understood. Please try a different command.")
                
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")

def perform_comparison(comparison_engine, viz_engine):
    """Perform direct comparison between two files"""
    try:
        with st.spinner("Comparing files..."):
            data1 = st.session_state.file1_data[st.session_state.selected_sheets['sheet1']]
            data2 = st.session_state.file2_data[st.session_state.selected_sheets['sheet2']]
            
            results = comparison_engine.compare_dataframes(data1, data2)
            st.session_state.comparison_results = results
            
            # Display comparison
            viz_engine.display_comparison_results(results, {
                'highlight_differences': True,
                'highlight_matches': True,
                'diff_color': 'yellow',
                'match_color': 'green'
            })
            
    except Exception as e:
        st.error(f"Error during comparison: {str(e)}")

def display_side_by_side_comparison():
    """Display two files side by side"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 File 1 Data")
        data1 = st.session_state.file1_data[st.session_state.selected_sheets['sheet1']]
        st.dataframe(data1, use_container_width=True, height=400)
        
        # Basic stats
        st.caption(f"Shape: {data1.shape[0]} rows × {data1.shape[1]} columns")
    
    with col2:
        st.subheader("📄 File 2 Data")
        data2 = st.session_state.file2_data[st.session_state.selected_sheets['sheet2']]
        st.dataframe(data2, use_container_width=True, height=400)
        
        # Basic stats
        st.caption(f"Shape: {data2.shape[0]} rows × {data2.shape[1]} columns")

def display_single_file_data():
    """Display single file data"""
    st.subheader("📄 File Data")
    data = st.session_state.file1_data[st.session_state.selected_sheets['sheet1']]
    
    # Display data with filtering options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search in data:", placeholder="Enter search term...")
    
    with col2:
        max_rows = st.number_input("Max rows to display:", min_value=10, max_value=1000, value=100)
    
    with col3:
        show_stats = st.checkbox("Show statistics", value=True)
    
    # Filter data if search term provided
    display_data = data.copy()
    if search_term:
        mask = display_data.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        display_data = display_data[mask]
    
    # Display dataframe
    st.dataframe(display_data.head(max_rows), use_container_width=True, height=400)
    
    if show_stats:
        # Display basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(data):,}")
        with col2:
            st.metric("Total Columns", f"{len(data.columns):,}")
        with col3:
            st.metric("Filtered Rows", f"{len(display_data):,}")
        with col4:
            missing_cells = data.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_cells:,}")

def export_comparison_results():
    """Export comparison results to Excel"""
    if st.session_state.comparison_results:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Comparison Report"):
                # Create Excel file with comparison results
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Write different sheets for different result types
                    if 'differences' in st.session_state.comparison_results:
                        st.session_state.comparison_results['differences'].to_excel(
                            writer, sheet_name='Differences', index=False
                        )
                    
                    if 'matches' in st.session_state.comparison_results:
                        st.session_state.comparison_results['matches'].to_excel(
                            writer, sheet_name='Matches', index=False
                        )
                    
                    if 'summary' in st.session_state.comparison_results:
                        summary_df = pd.DataFrame([st.session_state.comparison_results['summary']])
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                output.seek(0)
                
                # Create download button
                st.download_button(
                    label="📄 Download Excel Report",
                    data=output.getvalue(),
                    file_name="comparison_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            if st.button("📋 Copy Summary to Clipboard"):
                if 'summary' in st.session_state.comparison_results:
                    summary = st.session_state.comparison_results['summary']
                    summary_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
                    st.code(summary_text)
                    st.success("Summary displayed above - you can copy it manually")

def perform_multi_file_comparison(comparison_engine, viz_engine, selected_sheet):
    """Perform comparison across multiple files"""
    try:
        with st.spinner("Comparing multiple files..."):
            # Extract DataFrames for the selected sheet from all files
            dataframes = []
            for file_data in st.session_state.multi_file_data:
                if selected_sheet in file_data:
                    dataframes.append(file_data[selected_sheet])
            
            if len(dataframes) < 2:
                st.error("Not enough data to compare. Please ensure the selected sheet exists in all files.")
                return
            
            # Use file names as labels
            labels = st.session_state.uploaded_files if st.session_state.uploaded_files else None
            
            # Perform multi-file comparison
            results = comparison_engine.compare_multiple_dataframes(
                dataframes,
                labels=labels,
                method='exact'
            )
            
            st.session_state.multi_comparison_results = results
            st.success(f"✅ Successfully compared {len(dataframes)} files!")
            
    except Exception as e:
        st.error(f"Error during multi-file comparison: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

def export_multi_file_data(selected_sheet):
    """Export multi-file comparison data"""
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Export each file's data
            for i, (file_data, filename) in enumerate(zip(st.session_state.multi_file_data, st.session_state.uploaded_files)):
                if selected_sheet in file_data:
                    sheet_name = f"V{i+1}_{filename[:20]}"  # Limit sheet name length
                    file_data[selected_sheet].to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Export comparison results if available
            if st.session_state.multi_comparison_results:
                results = st.session_state.multi_comparison_results
                
                # Export change tracking
                if results['change_tracking']['changed_cells']:
                    changed_df = pd.DataFrame(results['change_tracking']['changed_cells'])
                    # Expand values for display
                    display_data = []
                    for _, row in changed_df.iterrows():
                        value_history = " → ".join([f"{v['value']}" for v in row['values']])
                        display_data.append({
                            'Row': row['row'],
                            'Column': row['column'],
                            'Change History': value_history
                        })
                    pd.DataFrame(display_data).to_excel(writer, sheet_name='Changes', index=False)
                
                # Export summary
                summary_data = []
                for key, value in results['summary'].items():
                    if key not in ['version_transitions', 'most_changed_columns', 'shapes']:
                        summary_data.append({'Metric': key, 'Value': str(value)})
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        
        st.download_button(
            label="📄 Download Multi-File Report",
            data=output.getvalue(),
            file_name="multi_file_comparison_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")

def display_multi_file_preview(selected_sheet):
    """Display preview of all uploaded files"""
    st.subheader("📄 File Previews")
    
    num_files = len(st.session_state.multi_file_data)
    cols = st.columns(min(num_files, 3))  # Max 3 columns
    
    for i, (file_data, filename) in enumerate(zip(st.session_state.multi_file_data, st.session_state.uploaded_files)):
        with cols[i % 3]:
            st.write(f"**{filename}**")
            if selected_sheet in file_data:
                df = file_data[selected_sheet]
                st.dataframe(df.head(5), use_container_width=True)
                st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            else:
                st.warning(f"Sheet '{selected_sheet}' not found")

if __name__ == "__main__":
    main()
