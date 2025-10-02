import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
import numpy as np

class VisualizationEngine:
    """Handle visualization and display of comparison results"""
    
    def __init__(self):
        self.color_scheme = {
            'yellow': '#FFEB3B',
            'green': '#4CAF50',
            'red': '#F44336',
            'blue': '#2196F3',
            'orange': '#FF9800',
            'difference': '#FFEB3B',
            'match': '#4CAF50'
        }
    
    def display_comparison_results(self, results: Dict[str, Any], query_params: Dict[str, Any]):
        """
        Display comprehensive comparison results
        
        Args:
            results: Comparison results from ComparisonEngine
            query_params: Parsed query parameters
        """
        # Display summary first
        self._display_summary_metrics(results['summary'])
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visual Comparison", 
            "📋 Detailed Results", 
            "📈 Statistics", 
            "🔍 Differences & Matches"
        ])
        
        with tab1:
            self._display_visual_comparison(results, query_params)
        
        with tab2:
            self._display_detailed_comparison(results)
        
        with tab3:
            self._display_statistics_charts(results)
        
        with tab4:
            self._display_differences_matches(results)
    
    def _display_summary_metrics(self, summary: Dict[str, Any]):
        """Display summary metrics in a dashboard-style layout"""
        st.subheader("📊 Comparison Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Cells",
                f"{summary['total_cells']:,}",
                help="Total number of cells compared"
            )
        
        with col2:
            st.metric(
                "Matching Cells",
                f"{summary['matching_cells']:,}",
                delta=f"{summary['match_percentage']:.1f}%",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "Different Cells",
                f"{summary['different_cells']:,}",
                delta=f"{100 - summary['match_percentage']:.1f}%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Common Columns",
                len(summary['common_columns']),
                help="Number of columns compared"
            )
        
        # Progress bar for match percentage
        st.progress(summary['match_percentage'] / 100)
        st.caption(f"Overall Match Rate: {summary['match_percentage']:.1f}%")
    
    def _display_visual_comparison(self, results: Dict[str, Any], query_params: Dict[str, Any]):
        """Display side-by-side visual comparison with highlighting"""
        st.subheader("🎨 Visual Comparison")
        
        if 'highlighted_data' in results and results['highlighted_data']:
            highlighted = results['highlighted_data']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**File 1 Data**")
                self._display_highlighted_dataframe(
                    highlighted['df1_data'],
                    highlighted['df1_highlight'],
                    query_params.get('diff_color', 'yellow'),
                    query_params.get('match_color', 'green')
                )
            
            with col2:
                st.write("**File 2 Data**")
                self._display_highlighted_dataframe(
                    highlighted['df2_data'],
                    highlighted['df2_highlight'],
                    query_params.get('diff_color', 'yellow'),
                    query_params.get('match_color', 'green')
                )
        
        else:
            # Fallback to basic comparison display
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**File 1 Data**")
                st.dataframe(results['df1_aligned'], use_container_width=True)
            
            with col2:
                st.write("**File 2 Data**")
                st.dataframe(results['df2_aligned'], use_container_width=True)
    
    def _display_highlighted_dataframe(self, data: pd.DataFrame, highlights: pd.DataFrame, 
                                     diff_color: str, match_color: str):
        """
        Display DataFrame with color highlighting
        
        Args:
            data: DataFrame with actual data
            highlights: DataFrame with highlighting information
            diff_color: Color for differences
            match_color: Color for matches
        """
        # Create styled DataFrame
        def highlight_cells(val, highlight_val, row_idx, col_name):
            """Apply highlighting based on highlight value"""
            if highlight_val == 'difference':
                return f'background-color: {self.color_scheme.get(diff_color, "#FFEB3B")}'
            elif highlight_val == 'match':
                return f'background-color: {self.color_scheme.get(match_color, "#4CAF50")}'
            return ''
        
        # Apply styling using Styler
        try:
            styled_df = data.style.apply(
                lambda x: [
                    highlight_cells(data.loc[idx, col], highlights.loc[idx, col], idx, col)
                    for idx, col in zip(x.index, [x.name] * len(x))
                ],
                axis=0
            )
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
        except Exception as e:
            # Fallback to regular dataframe if styling fails
            st.dataframe(data, use_container_width=True, height=400)
            st.caption("Note: Highlighting not available due to technical limitations")
    
    def _display_detailed_comparison(self, results: Dict[str, Any]):
        """Display detailed comparison matrix and data"""
        st.subheader("🔍 Detailed Comparison Matrix")
        
        # Display comparison matrix
        comparison_matrix = results['comparison_matrix']
        
        # Create heatmap of comparison results
        fig = px.imshow(
            comparison_matrix.astype(int),
            labels=dict(x="Columns", y="Rows", color="Match"),
            title="Comparison Heatmap (Green=Match, Red=Difference)",
            color_continuous_scale=["red", "green"]
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display raw comparison matrix
        with st.expander("📋 Raw Comparison Matrix"):
            st.dataframe(comparison_matrix, use_container_width=True)
    
    def _display_statistics_charts(self, results: Dict[str, Any]):
        """Display statistical charts and analysis"""
        st.subheader("📈 Statistical Analysis")
        
        summary = results['summary']
        
        # Column-wise statistics
        if 'column_statistics' in summary:
            col_stats = summary['column_statistics']
            
            # Create bar chart for column-wise match rates
            columns = list(col_stats.keys())
            match_percentages = [col_stats[col]['match_percentage'] for col in columns]
            
            fig_bar = px.bar(
                x=columns,
                y=match_percentages,
                title="Match Percentage by Column",
                labels={'x': 'Columns', 'y': 'Match Percentage (%)'},
                color=match_percentages,
                color_continuous_scale="RdYlGn"
            )
            
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Detailed column statistics table
            col_stats_df = pd.DataFrame.from_dict(col_stats, orient='index')
            col_stats_df = col_stats_df.round(2)
            
            st.subheader("📊 Column-wise Statistics")
            st.dataframe(col_stats_df, use_container_width=True)
        
        # Overall distribution pie chart
        fig_pie = px.pie(
            values=[summary['matching_cells'], summary['different_cells']],
            names=['Matches', 'Differences'],
            title="Overall Distribution of Matches vs Differences",
            color_discrete_map={'Matches': '#4CAF50', 'Differences': '#FFEB3B'}
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    def _display_differences_matches(self, results: Dict[str, Any]):
        """Display detailed differences and matches tables"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("❌ Differences")
            if not results['differences'].empty:
                # Filter and display differences
                diff_display = results['differences'].copy()
                
                # Add search functionality
                search_diff = st.text_input("🔍 Search differences:", key="search_diff")
                if search_diff:
                    mask = diff_display.astype(str).apply(
                        lambda x: x.str.contains(search_diff, case=False, na=False)
                    ).any(axis=1)
                    diff_display = diff_display[mask]
                
                st.dataframe(diff_display, use_container_width=True, height=400)
                st.caption(f"Showing {len(diff_display)} of {len(results['differences'])} differences")
            else:
                st.info("No differences found between the files!")
        
        with col2:
            st.subheader("✅ Matches")
            if not results['matches'].empty:
                # Filter and display matches
                match_display = results['matches'].copy()
                
                # Add search functionality
                search_match = st.text_input("🔍 Search matches:", key="search_match")
                if search_match:
                    mask = match_display.astype(str).apply(
                        lambda x: x.str.contains(search_match, case=False, na=False)
                    ).any(axis=1)
                    match_display = match_display[mask]
                
                st.dataframe(match_display, use_container_width=True, height=400)
                st.caption(f"Showing {len(match_display)} of {len(results['matches'])} matches")
            else:
                st.info("No matches found between the files.")
    
    def display_analysis_results(self, results: Dict[str, Any]):
        """
        Display single file analysis results
        
        Args:
            results: Analysis results from ComparisonEngine
        """
        st.subheader("📊 File Analysis Results")
        
        # Basic information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{results['shape'][0]:,}")
        with col2:
            st.metric("Columns", f"{results['shape'][1]:,}")
        with col3:
            st.metric("Duplicate Rows", f"{results['duplicate_rows']:,}")
        with col4:
            total_missing = sum(results['missing_values'].values())
            st.metric("Missing Values", f"{total_missing:,}")
        
        # Create tabs for different analysis views
        tab1, tab2, tab3 = st.tabs(["📈 Summary Statistics", "📋 Data Types", "🔍 Missing Values"])
        
        with tab1:
            if results['summary_stats']:
                st.subheader("Summary Statistics")
                stats_df = pd.DataFrame(results['summary_stats'])
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("No numeric columns found for statistical analysis.")
        
        with tab2:
            st.subheader("Data Types and Unique Values")
            dtype_df = pd.DataFrame({
                'Column': results['columns'],
                'Data Type': [results['dtypes'][col] for col in results['columns']],
                'Unique Values': [results['unique_values_per_column'][col] for col in results['columns']]
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with tab3:
            st.subheader("Missing Values Analysis")
            missing_df = pd.DataFrame(list(results['missing_values'].items()), 
                                    columns=['Column', 'Missing Count'])
            missing_df['Missing Percentage'] = (missing_df['Missing Count'] / results['shape'][0] * 100).round(2)
            
            # Create bar chart for missing values
            if missing_df['Missing Count'].sum() > 0:
                fig = px.bar(missing_df, x='Column', y='Missing Count',
                           title="Missing Values by Column",
                           color='Missing Percentage',
                           color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found in the dataset!")
            
            st.dataframe(missing_df, use_container_width=True)
    
    def create_download_link(self, df: pd.DataFrame, filename: str, link_text: str) -> str:
        """
        Create a download link for DataFrame
        
        Args:
            df: DataFrame to download
            filename: Name for downloaded file
            link_text: Text for download link
            
        Returns:
            HTML string for download link
        """
        import io
        import base64
        
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    
    def display_multi_file_comparison(self, results: Dict[str, Any]):
        """
        Display comparison results for multiple files (3+)
        
        Args:
            results: Multi-file comparison results from ComparisonEngine
        """
        st.header("📊 Multi-File Comparison Results")
        
        # Display overall summary
        self._display_multi_file_summary(results['summary'])
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔄 Version Timeline",
            "📋 Pairwise Comparisons", 
            "🎯 Change Tracking",
            "📈 Statistics"
        ])
        
        with tab1:
            self._display_version_timeline(results)
        
        with tab2:
            self._display_pairwise_comparisons(results)
        
        with tab3:
            self._display_change_tracking(results)
        
        with tab4:
            self._display_multi_file_statistics(results)
    
    def _display_multi_file_summary(self, summary: Dict[str, Any]):
        """Display summary metrics for multi-file comparison"""
        st.subheader("📊 Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Versions Compared",
                summary['num_versions'],
                help="Number of file versions analyzed"
            )
        
        with col2:
            st.metric(
                "Total Cells",
                f"{summary['total_cells_per_version']:,}",
                help="Total cells per version"
            )
        
        with col3:
            st.metric(
                "Changed Cells",
                f"{summary['total_changed_cells']:,}",
                help="Cells that changed across versions"
            )
        
        with col4:
            st.metric(
                "Unchanged Cells",
                f"{summary['total_unchanged_cells']:,}",
                help="Cells that remained constant"
            )
        
        with col5:
            st.metric(
                "Change Rate",
                f"{summary['change_percentage']:.1f}%",
                help="Percentage of cells that changed"
            )
        
        # Progress bar for change rate
        st.progress(summary['change_percentage'] / 100)
        st.caption(f"Tracking changes across: {', '.join(summary['version_labels'])}")
    
    def _display_version_timeline(self, results: Dict[str, Any]):
        """Display timeline view of changes across versions"""
        st.subheader("🔄 Version Change Timeline")
        
        summary = results['summary']
        
        # Create timeline visualization
        timeline_data = []
        for transition in summary['version_transitions']:
            timeline_data.append({
                'Transition': transition['transition'],
                'Differences': transition['differences'],
                'Match %': transition['match_percentage']
            })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            
            # Create line chart showing match percentage over transitions
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timeline_df['Transition'],
                y=timeline_df['Match %'],
                mode='lines+markers',
                name='Match Percentage',
                line=dict(color='green', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="Match Percentage Across Version Transitions",
                xaxis_title="Version Transition",
                yaxis_title="Match Percentage (%)",
                height=400,
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display transition details table
            st.subheader("📋 Transition Details")
            st.dataframe(timeline_df, use_container_width=True)
        else:
            st.info("No version transitions to display")
    
    def _display_pairwise_comparisons(self, results: Dict[str, Any]):
        """Display pairwise comparison results"""
        st.subheader("📋 Pairwise Version Comparisons")
        
        pairwise = results['pairwise_comparisons']
        
        if not pairwise:
            st.info("No pairwise comparisons available")
            return
        
        # Create selector for which comparison to view
        comparison_keys = list(pairwise.keys())
        selected_comparison = st.selectbox(
            "Select comparison to view:",
            comparison_keys,
            key="pairwise_selector"
        )
        
        if selected_comparison:
            comparison_result = pairwise[selected_comparison]
            
            # Display this comparison using standard comparison display
            st.subheader(f"Comparing: {selected_comparison}")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cells", f"{comparison_result['summary']['total_cells']:,}")
            with col2:
                st.metric("Differences", f"{comparison_result['summary']['different_cells']:,}")
            with col3:
                st.metric("Match %", f"{comparison_result['summary']['match_percentage']:.1f}%")
            
            # Display differences
            if not comparison_result['differences'].empty:
                st.subheader("❌ Differences Found")
                st.dataframe(comparison_result['differences'], use_container_width=True, height=300)
            else:
                st.success("No differences found in this comparison!")
    
    def _display_change_tracking(self, results: Dict[str, Any]):
        """Display detailed change tracking across all versions"""
        st.subheader("🎯 Change Tracking Across All Versions")
        
        change_tracking = results['change_tracking']
        
        # Display most frequently changed columns
        st.subheader("🔥 Most Changed Columns")
        if change_tracking['column_change_frequency']:
            col_changes = pd.DataFrame(
                [(col, changes) for col, changes in change_tracking['column_change_frequency'].items()],
                columns=['Column', 'Number of Changes']
            ).sort_values('Number of Changes', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                col_changes.head(10),
                x='Column',
                y='Number of Changes',
                title="Top 10 Most Changed Columns",
                color='Number of Changes',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.dataframe(col_changes, use_container_width=True)
        
        # Display changed cells with history
        st.subheader("📜 Cell Change History")
        
        if change_tracking['changed_cells']:
            # Create a searchable/filterable view of changed cells
            changed_df = pd.DataFrame(change_tracking['changed_cells'])
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                search_column = st.text_input("🔍 Filter by column name:", key="change_col_filter")
            with col2:
                max_rows_display = st.number_input("Max rows to display:", min_value=10, max_value=1000, value=100, key="change_rows")
            
            # Apply filters
            if search_column:
                changed_df = changed_df[changed_df['column'].astype(str).str.contains(search_column, case=False, na=False)]
            
            # Expand values column for display
            display_data = []
            for _, row in changed_df.head(max_rows_display).iterrows():
                value_history = " → ".join([f"{v['version']}: {v['value']}" for v in row['values']])
                display_data.append({
                    'Row': row['row'],
                    'Column': row['column'],
                    'Change History': value_history
                })
            
            if display_data:
                st.dataframe(pd.DataFrame(display_data), use_container_width=True, height=400)
                st.caption(f"Showing {len(display_data)} of {len(change_tracking['changed_cells'])} changed cells")
            else:
                st.info("No changes match your filters")
        else:
            st.success("No cells changed across all versions!")
    
    def _display_multi_file_statistics(self, results: Dict[str, Any]):
        """Display statistical analysis for multi-file comparison"""
        st.subheader("📈 Multi-File Statistics")
        
        summary = results['summary']
        
        # Overall distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of changed vs unchanged
            fig_pie = px.pie(
                values=[summary['total_changed_cells'], summary['total_unchanged_cells']],
                names=['Changed', 'Unchanged'],
                title="Overall Cell Change Distribution",
                color_discrete_map={'Changed': '#FF9800', 'Unchanged': '#4CAF50'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart of differences per transition
            if summary['version_transitions']:
                transition_data = pd.DataFrame([
                    {'Transition': t['transition'], 'Differences': t['differences']}
                    for t in summary['version_transitions']
                ])
                
                fig_bar = px.bar(
                    transition_data,
                    x='Transition',
                    y='Differences',
                    title="Differences Per Version Transition",
                    color='Differences',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Version shapes table
        st.subheader("📐 Version Shapes")
        shapes_data = []
        for label, shape in summary['shapes'].items():
            shapes_data.append({
                'Version': label,
                'Rows': shape[0],
                'Columns': shape[1],
                'Total Cells': shape[0] * shape[1]
            })
        
        st.dataframe(pd.DataFrame(shapes_data), use_container_width=True)
        
        # Most changed columns summary
        if summary['most_changed_columns']:
            st.subheader("🎯 Top 5 Most Changed Columns")
            top_changes = pd.DataFrame(
                summary['most_changed_columns'],
                columns=['Column', 'Changes']
            )
            st.dataframe(top_changes, use_container_width=True)
