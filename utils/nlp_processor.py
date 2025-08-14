import re
from typing import Dict, List, Any, Optional
import pandas as pd

class NLPProcessor:
    """Process natural language queries for data analysis and comparison"""
    
    def __init__(self):
        # Define command patterns and their corresponding actions
        self.command_patterns = {
            'compare': [
                r'compare.*(?:between|and)',
                r'compare.*(?:files?|datasets?|sheets?)',
                r'compare.*(?:two|both)',
                r'find.*(?:differences?|matches?)',
                r'highlight.*(?:differences?|matches?)'
            ],
            'analyze': [
                r'analyze.*(?:file|data|sheet)',
                r'show.*(?:statistics?|stats)',
                r'summarize.*(?:data|file)',
                r'describe.*(?:data|dataset)'
            ],
            'filter': [
                r'filter.*(?:rows?|data|where)',
                r'show.*(?:rows?|data).*where',
                r'find.*(?:rows?|records?).*(?:where|with)'
            ],
            'search': [
                r'search.*(?:for|in)',
                r'find.*(?:cells?|values?)',
                r'look.*(?:for|up)'
            ]
        }
        
        # Color mapping
        self.color_mapping = {
            'yellow': ['yellow', 'amber', 'gold'],
            'green': ['green', 'lime', 'emerald'],
            'red': ['red', 'crimson', 'scarlet'],
            'blue': ['blue', 'azure', 'navy'],
            'orange': ['orange', 'tangerine']
        }
        
        # Highlight type mapping
        self.highlight_types = {
            'differences': ['difference', 'diff', 'different', 'mismatch', 'unlike'],
            'matches': ['match', 'same', 'identical', 'equal', 'similar']
        }
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query and extract intent and parameters
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary with parsed query information
        """
        query_lower = query.lower().strip()
        
        # Initialize result structure
        parsed_query = {
            'action': 'unknown',
            'highlight_differences': False,
            'highlight_matches': False,
            'diff_color': 'yellow',
            'match_color': 'green',
            'columns': [],
            'conditions': [],
            'search_terms': [],
            'comparison_type': 'exact',
            'original_query': query
        }
        
        # Determine main action
        parsed_query['action'] = self._identify_action(query_lower)
        
        # Extract highlighting preferences
        highlighting = self._extract_highlighting_info(query_lower)
        parsed_query.update(highlighting)
        
        # Extract column references
        parsed_query['columns'] = self._extract_columns(query_lower)
        
        # Extract search terms
        parsed_query['search_terms'] = self._extract_search_terms(query_lower)
        
        # Extract conditions for filtering
        parsed_query['conditions'] = self._extract_conditions(query_lower)
        
        # Determine comparison type
        parsed_query['comparison_type'] = self._determine_comparison_type(query_lower)
        
        return parsed_query
    
    def _identify_action(self, query: str) -> str:
        """
        Identify the main action from the query
        
        Args:
            query: Lowercase query string
            
        Returns:
            Action type as string
        """
        for action, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return action
        
        return 'analyze'  # Default action
    
    def _extract_highlighting_info(self, query: str) -> Dict[str, Any]:
        """
        Extract highlighting preferences from query
        
        Args:
            query: Lowercase query string
            
        Returns:
            Dictionary with highlighting preferences
        """
        highlighting = {
            'highlight_differences': False,
            'highlight_matches': False,
            'diff_color': 'yellow',
            'match_color': 'green'
        }
        
        # Check for highlighting keywords
        for highlight_type, keywords in self.highlight_types.items():
            for keyword in keywords:
                if keyword in query:
                    if highlight_type == 'differences':
                        highlighting['highlight_differences'] = True
                    elif highlight_type == 'matches':
                        highlighting['highlight_matches'] = True
        
        # Extract color preferences
        colors_found = self._extract_colors(query)
        
        # Assign colors based on context
        if colors_found:
            # Try to match colors with context
            diff_patterns = r'(?:difference|diff|different|mismatch).*?(?:in|with)\s+(\w+)'
            match_patterns = r'(?:match|same|identical|equal).*?(?:in|with)\s+(\w+)'
            
            diff_color_match = re.search(diff_patterns, query)
            match_color_match = re.search(match_patterns, query)
            
            if diff_color_match and diff_color_match.group(1) in colors_found:
                highlighting['diff_color'] = diff_color_match.group(1)
            elif colors_found:
                # If specific context not found, use first color for differences
                highlighting['diff_color'] = colors_found[0]
            
            if match_color_match and match_color_match.group(1) in colors_found:
                highlighting['match_color'] = match_color_match.group(1)
            elif len(colors_found) > 1:
                highlighting['match_color'] = colors_found[1]
        
        return highlighting
    
    def _extract_colors(self, query: str) -> List[str]:
        """
        Extract color names from query
        
        Args:
            query: Lowercase query string
            
        Returns:
            List of color names found
        """
        colors_found = []
        
        for color, variants in self.color_mapping.items():
            for variant in variants:
                if variant in query:
                    colors_found.append(color)
                    break
        
        return colors_found
    
    def _extract_columns(self, query: str) -> List[str]:
        """
        Extract column references from query
        
        Args:
            query: Lowercase query string
            
        Returns:
            List of column references
        """
        columns = []
        
        # Look for column patterns
        column_patterns = [
            r'column\s+([a-zA-Z]\w*)',
            r'field\s+([a-zA-Z]\w*)',
            r'(?:in|from)\s+([a-zA-Z]\w*)\s+column',
            r'(?:where|and)\s+([a-zA-Z]\w*)\s+(?:is|equals?|contains?)'
        ]
        
        for pattern in column_patterns:
            matches = re.findall(pattern, query)
            columns.extend(matches)
        
        return list(set(columns))  # Remove duplicates
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """
        Extract search terms from query
        
        Args:
            query: Lowercase query string
            
        Returns:
            List of search terms
        """
        search_terms = []
        
        # Look for quoted strings
        quoted_terms = re.findall(r'"([^"]*)"', query)
        search_terms.extend(quoted_terms)
        
        # Look for search patterns
        search_patterns = [
            r'search\s+for\s+([^\s]+)',
            r'find\s+([^\s]+)',
            r'contains?\s+([^\s]+)',
            r'equals?\s+([^\s]+)'
        ]
        
        for pattern in search_patterns:
            matches = re.findall(pattern, query)
            search_terms.extend(matches)
        
        return list(set(search_terms))
    
    def _extract_conditions(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract filtering conditions from query
        
        Args:
            query: Lowercase query string
            
        Returns:
            List of condition dictionaries
        """
        conditions = []
        
        # Simple condition patterns
        condition_patterns = [
            r'where\s+(\w+)\s+(equals?|is|contains?|>|<|>=|<=)\s+([^\s]+)',
            r'(\w+)\s+(equals?|is|contains?|>|<|>=|<=)\s+([^\s]+)'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if len(match) == 3:
                    conditions.append({
                        'column': match[0],
                        'operator': match[1],
                        'value': match[2]
                    })
        
        return conditions
    
    def _determine_comparison_type(self, query: str) -> str:
        """
        Determine the type of comparison to perform
        
        Args:
            query: Lowercase query string
            
        Returns:
            Comparison type string
        """
        if any(word in query for word in ['fuzzy', 'similar', 'approximate']):
            return 'fuzzy'
        elif any(word in query for word in ['numeric', 'threshold', 'tolerance']):
            return 'numeric_threshold'
        else:
            return 'exact'
    
    def generate_query_suggestions(self, available_columns: List[str]) -> List[str]:
        """
        Generate query suggestions based on available columns
        
        Args:
            available_columns: List of available column names
            
        Returns:
            List of suggested queries
        """
        suggestions = [
            "Compare these two files and highlight differences in yellow, matches in green",
            "Show me all rows where values are different between files",
            "Find matching values in both datasets",
            "Highlight cells that don't match between sheets",
            "Compare data and show summary statistics",
            "Analyze this file and show basic statistics"
        ]
        
        # Add column-specific suggestions if columns are available
        if available_columns:
            sample_column = available_columns[0] if available_columns else "column_name"
            suggestions.extend([
                f"Find differences in {sample_column} column",
                f"Show rows where {sample_column} is different",
                f"Compare {sample_column} values between files"
            ])
        
        return suggestions
    
    def validate_query(self, query: str, available_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate if query can be executed with available data
        
        Args:
            query: Query string
            available_data: Information about available data
            
        Returns:
            Validation result dictionary
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        parsed = self.parse_query(query)
        
        # Check if comparison is requested but only one file available
        if parsed['action'] == 'compare' and available_data.get('file_count', 0) < 2:
            validation['valid'] = False
            validation['errors'].append("Comparison requires two files, but only one file is available")
        
        # Check if referenced columns exist
        if parsed['columns'] and 'available_columns' in available_data:
            available_cols = available_data['available_columns']
            missing_columns = [col for col in parsed['columns'] if col not in available_cols]
            
            if missing_columns:
                validation['warnings'].append(f"Referenced columns not found: {', '.join(missing_columns)}")
                validation['suggestions'].append(f"Available columns: {', '.join(available_cols[:5])}")
        
        return validation
