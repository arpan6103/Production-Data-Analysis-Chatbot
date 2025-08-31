from dotenv import load_dotenv
import requests
import re
import time
import pandas as pd
import statistics
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any
import os
import json

load_dotenv()

class ProductionDataChatbot:
    def __init__(self, csv_file_path='data1.csv'):
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        self.csv_file_path = csv_file_path

        self.conversation_history = []  # Store conversation context
        self.current_session = {
            'last_filtered_df': None,
            'last_query_analysis': None,
            'active_time_filters': {},
            'active_metric_filters': {},
            'context_expires_after': 5  # Number of exchanges before context expires
        }
        
        # Load CSV data
        try:
            self.df = pd.read_csv(csv_file_path)
            print(f"✅ Successfully loaded {len(self.df)} records from {csv_file_path}")
            print(f"📊 Columns: {list(self.df.columns)}")
        except FileNotFoundError:
            print(f"❌ CSV file not found: {csv_file_path}")
            raise
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise
        
        # Ensure Date column is datetime
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Pre-calculate metrics for consistent filtering
        self.pre_calculate_metrics()
        
        # Model configuration
        self.model = "google/gemini-2.5-flash-lite"
        
        # Production data columns
        self.key_metrics = [
            "Date", "Target", "Production", "Rejection", "Downtime", 
            "Run Time", "QualityRate", "Availibility", "Performance Efficiency", "OEE"
        ]

        self.NUMERIC_FIELDS = ["Downtime", "OEE", "Rejection", "QualityRate", "Production", "Target", "Run Time","Performance Efficiency","Availibility"]

    def pre_calculate_metrics(self):
        """Pre-calculate all metrics in the DataFrame for consistent filtering"""
        print("🧮 Pre-calculating metrics for smart filtering...")
        
        def safe_divide(a, b):
            return a / b if b > 0 else 0
        
        # Calculate Performance Efficiency
        self.df['Calculated_Performance_Efficiency'] = (
            (self.df['Production'].fillna(0) + self.df['Rejection'].fillna(0)) / 
            self.df['Target'].fillna(1)  # Avoid division by zero
        ).fillna(0)
        
        # Calculate Quality Rate
        total_produced = self.df['Production'].fillna(0) + self.df['Rejection'].fillna(0)
        self.df['Calculated_QualityRate'] = (
            self.df['Production'].fillna(0) / total_produced
        ).fillna(0)
        
        # Calculate Availability
        total_time = self.df['Run Time'].fillna(0) + self.df['Downtime'].fillna(0)
        self.df['Calculated_Availability'] = (
            self.df['Run Time'].fillna(0) / total_time
        ).fillna(0)
        
        # Calculate OEE
        self.df['Calculated_OEE'] = (
            self.df['Calculated_Availability'] * 
            self.df['Calculated_Performance_Efficiency'] * 
            self.df['Calculated_QualityRate']
        ).fillna(0)
        
        # Update the original columns with calculated values
        self.df['OEE'] = self.df['Calculated_OEE']
        self.df['QualityRate'] = self.df['Calculated_QualityRate'] 
        self.df['Availibility'] = self.df['Calculated_Availability']  # Note: keeping original column name typo
        self.df['Performance Efficiency'] = self.df['Calculated_Performance_Efficiency']
        
        print(f"✅ Calculated metrics for {len(self.df)} records")
        print(f"📊 OEE range: {self.df['OEE'].min():.4f} - {self.df['OEE'].max():.4f}")
        print(f"📊 Records with OEE > 0.7: {len(self.df[self.df['OEE'] > 0.7])}")

    def analyze_query(self, query: str, available_columns: List[str]) -> Dict[str, Any]:
        """Enhanced query analysis with conversation context"""
        
        # Check if this is a follow-up question
        if self.is_followup_question(query):
            print("🔗 Detected follow-up question - using conversation context")
            return self.analyze_followup_query(query, available_columns)
        else:
            # Clear context if it's a completely new topic
            # if self.should_reset_context(query):
            #     print("🆕 New topic detected - resetting conversation context")
            #     self.reset_conversation_context()
            
            print("🎯 Analyzing fresh query")
            return self.analyze_fresh_query(query, available_columns)
    
    def analyze_fresh_query(self, query: str, available_columns: List[str]) -> Dict[str, Any]:
        """Analyze new/fresh questions (your existing analyze_query logic)"""
        print(available_columns)
        system_prompt = f"""
        You are a query analysis expert for manufacturing production data. Your job is to analyze user queries and extract structured filtering criteria.

        AVAILABLE DATA COLUMNS: {', '.join(available_columns)}

        EXTRACT THE FOLLOWING INFORMATION FROM THE USER QUERY:

        1. **DATE_FILTERS**: Extract any date-related criteria
        - Specific dates (e.g., "2024-01-15")
        - Date ranges (e.g., "January to March 2024")  
        - Day patterns (e.g., "all Sundays", "weekends", "Mondays")
        - Relative dates (e.g., "last week", "past month")
        - Quarters (e.g., "Q1 2024")
        - Multiple periods (e.g., "January and February", "Q1 and Q2")

        2. **METRIC_FILTERS**: Extract any metric-based filtering criteria
        - Thresholds (e.g., "OEE > 0.85", "downtime < 2 hours")
        - Comparisons (e.g., "highest production", "lowest quality rate")
        - Specific values (e.g., "rejection = 0")
        - Top/Bottom N (e.g., "top 5 days", "worst 10 records")

        3. **INTENT**: Determine what the user wants to do
        - "analyze" - General analysis/summary
        - "compare" - Compare periods/metrics
        - "list" - Show specific records
        - "calculate" - Calculate averages/totals
        - "trend" - Show trends over time
        - "count" - Count records matching criteria

        4. **OUTPUT_FORMAT**: Determine preferred output format
        - "summary" - Brief summary with key insights
        - "detailed" - Detailed breakdown with all data
        - "table" - Tabular format
        - "chart_data" - Data suitable for charts
        - "comparison" - Side-by-side comparison format

        5. **ANALYSIS_TYPE**: Determine if user wants combined or separate analysis for multiple periods
        - "combined" - Aggregate data across all periods (e.g., "total for January and February", quater, q1)
        - "separate" - Show each period individually (e.g., "January and February separately")
        - "single" - Only one period involved

        6. **METRICS_OF_INTEREST**: Which specific metrics the user cares about
        - From available columns: {', '.join(available_columns)}

        7. **MULTI_PERIOD**: Detect if query involves multiple time periods
        - true/false based on whether multiple months, quarters, or date ranges are mentioned

        RESPONSE FORMAT (JSON):
        {{
            "date_filters": {{
                "specific_dates": ["2024-01-15", "2024-02-20"],
                "date_ranges": [{{"start": "2024-01-01", "end": "2024-03-31"}}],
                "day_patterns": ["sunday", "monday"],
                "relative_periods": ["last_month", "this_quarter"],
                "quarters": ["Q1_2024"],
                "months": ["january_2024", "february_2024"]
            }},
            "metric_filters": {{
                "thresholds": [{{"metric": "OEE", "operator": ">", "value": 0.85}}],
                "comparisons": [{{"type": "maximum", "metric": "Production"}}],
                "top_n": [{{"n": 5, "metric": "Quality_Rate", "direction": "highest"}}],
                "exact_values": [{{"metric": "Rejection", "value": 0}}]
            }},
            "intent": "analyze",
            "output_format": "summary",
            "analysis_type": "separate",
            "multi_period": true,
            "metrics_of_interest": ["OEE", "Quality_Rate", "Production"],
            "confidence": 0.95
        }}

        CRITICAL ANALYSIS_TYPE DETECTION:
        - Look for keywords like "separately", "individually", "each", "respectively", "compare"
        - If found with multiple periods → "separate"
        - Look for keywords like "combined", "total", "overall", "together", "aggregate"
        - If found with multiple periods → "combined"  
        - If only one period mentioned → "single"
        - Default for multiple periods without clear indicators → "separate"
        - Always put combined when the user asks for quarter or somehting like q1

        IMPORTANT RULES:
        - Only include filters/criteria that are explicitly mentioned in the query
        - If no specific criteria is mentioned, leave arrays empty
        - For day patterns like "Sundays", extract as ["sunday"]
        - For quarters, use format "Q1_2024", "Q2_2024", etc.
        - For months, use format "january_2024", "february_2024", etc.
        - Set confidence between 0.0-1.0 based on clarity of the query
        - If query is ambiguous, set confidence lower
        - Pay special attention to "separately", "individually" → analysis_type: "separate"
        """

        # Make the LLM API call
        headers = {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "max_tokens": 1500,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this query: '{query}'"}
            ]
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                
                # Extract JSON from the response
                try:
                    json_match = re.search(r'\{.*\}', result, re.DOTALL)
                    if json_match:
                        query_analysis = json.loads(json_match.group())
                        return query_analysis
                    else:
                        print("⚠️ No JSON found in LLM response")
                        return self._get_fallback_analysis(query)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON parsing error: {e}")
                    return self._get_fallback_analysis(query)
            else:
                print(f"❌ LLM API Error: {response.status_code}")
                return self._get_fallback_analysis(query)
                
        except Exception as e:
            print(f"❌ Error in query analysis: {e}")
            return self._get_fallback_analysis(query)

    def build_context_prompt_for_followup(self):
        """Build context prompt from conversation history"""
        if not self.conversation_history:
            return ""
        
        context_lines = ["RECENT CONVERSATION:"]
        for i, entry in enumerate(self.conversation_history[-2:], 1):  # Last 2 exchanges
            context_lines.append(f"Exchange {i}:")
            context_lines.append(f"  User asked: '{entry['user_query']}'")
            context_lines.append(f"  Analyzed intent: {entry['query_analysis'].get('intent', 'unknown')}")
            context_lines.append(f"  Time filters used: {entry['query_analysis'].get('date_filters', {})}")
            context_lines.append(f"  Records found: {entry['filtered_df_summary']['row_count']}")
            context_lines.append("")
        
        return "\n".join(context_lines)

    def analyze_followup_query(self, query: str, available_columns: List[str]) -> Dict[str, Any]:
        """Analyze follow-up questions with context inheritance"""
        
        if not self.current_session['last_query_analysis']:
            print("⚠️ No previous context found, treating as fresh query")
            return self.analyze_fresh_query(query, available_columns)
        
        # Start with the last query analysis as base
        base_analysis = self.current_session['last_query_analysis'].copy()
        
        # Build enhanced system prompt with conversation context
        context_prompt = self.build_context_prompt_for_followup()
        
        # Enhanced system prompt for follow-up analysis
        system_prompt = f"""
        You are a query analysis expert for manufacturing production data. You're analyzing a FOLLOW-UP question that refers to a previous conversation.

        AVAILABLE DATA COLUMNS: {', '.join(available_columns)}

        {context_prompt}

        FOLLOWUP ANALYSIS RULES:
        1. This is a follow-up question to previous queries
        2. INHERIT time filters from previous context unless user explicitly mentions new dates/periods
        3. INHERIT metric filters from previous context unless user explicitly overrides them
        4. If user mentions new metrics, ADD them to existing context (don't replace)
        5. If user asks for "more details", "break it down", "show daily data" → change output_format to "detailed"
        6. If user asks "what about [metric]?" → inherit all previous filters, just change metrics_of_interest
        7. If user asks comparative questions ("compare", "versus") → set analysis_type to "separate"
        8. If user says "same period", "that time" → use exact same date_filters as before

        CONTEXT TO INHERIT:
        - Previous date filters: {base_analysis.get('date_filters', {})}
        - Previous metric filters: {base_analysis.get('metric_filters', {})}
        - Previous intent: {base_analysis.get('intent', 'analyze')}
        - Previous analysis_type: {base_analysis.get('analysis_type', 'single')}
        - Previous multi_period: {base_analysis.get('multi_period', False)}

        FOLLOWUP DETECTION PATTERNS:
        - "that" / "those" / "it" → refers to previous results
        - "same period" → use exact same date_filters
        - "what about X?" → inherit filters, change metrics to X
        - "more details" → inherit everything, change output_format to "detailed"
        - "compare to Y" → inherit base filters, add comparison
        - "show daily" / "break it down" → inherit filters, change output_format to "detailed"

        Your task: Analyze the follow-up query and return MODIFIED analysis that inherits appropriate context.

        RESPONSE FORMAT (JSON):
        {{
            "date_filters": {{
                "specific_dates": [],
                "date_ranges": [],
                "day_patterns": [],
                "relative_periods": [],
                "quarters": [],
                "months": []
            }},
            "metric_filters": {{
                "thresholds": [],
                "comparisons": [],
                "top_n": [],
                "exact_values": []
            }},
            "intent": "analyze",
            "output_format": "summary",
            "analysis_type": "single",
            "multi_period": false,
            "metrics_of_interest": [],
            "confidence": 0.95,
            "context_inherited": true,
            "modifications_made": ["inherited_date_filters", "changed_metrics", "updated_output_format"]
        }}

        CRITICAL: If the follow-up doesn't mention time periods, inherit ALL date_filters from context.
        If the follow-up doesn't mention thresholds, inherit metric_filters from context.
        """
        headers = {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "max_tokens": 1500,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this follow-up query: '{query}'"}
            ]
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                
                # Extract JSON from the response
                try:
                    json_match = re.search(r'\{.*\}', result, re.DOTALL)
                    if json_match:
                        followup_analysis = json.loads(json_match.group())
                        
                        # Apply intelligent inheritance fallback if LLM didn't inherit properly
                        enhanced_analysis = self.apply_context_inheritance_fallback(
                            base_analysis, followup_analysis, query
                        )
                        
                        print(f"🔗 Context inherited: {enhanced_analysis.get('modifications_made', [])}")
                        return enhanced_analysis
                    else:
                        print("⚠️ No JSON found in LLM response for follow-up")
                        return self.apply_simple_inheritance(base_analysis, query)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON parsing error in follow-up analysis: {e}")
                    return self.apply_simple_inheritance(base_analysis, query)
            else:
                print(f"❌ LLM API Error in follow-up analysis: {response.status_code}")
                return self.apply_simple_inheritance(base_analysis, query)
                
        except Exception as e:
            print(f"❌ Error in follow-up analysis: {e}")
            return self.apply_simple_inheritance(base_analysis, query)

    def apply_context_inheritance_fallback(self, base_analysis, followup_analysis, query):
        """Apply intelligent context inheritance if LLM missed something"""
        
        enhanced = followup_analysis.copy()
        modifications = enhanced.get('modifications_made', [])

        # Inherit metrics_of_interest if missing/empty
        if not followup_analysis.get("metrics_of_interest"):
            followup_analysis["metrics_of_interest"] = base_analysis.get("metrics_of_interest", [])
            followup_analysis.setdefault("modifications_made", []).append("inherited_metrics_of_interest")
        
        # If no date filters in follow-up but base had them, inherit them
        if (not any(followup_analysis['date_filters'].values()) and 
            any(base_analysis['date_filters'].values()) and
            not self.has_explicit_time_reference(query)):
            
            enhanced['date_filters'] = base_analysis['date_filters'].copy()
            modifications.append("inherited_date_filters_fallback")
        
        # If no metric filters in follow-up but base had them, inherit them
        if (not any(followup_analysis['metric_filters'].values()) and 
            any(base_analysis['metric_filters'].values()) and
            not self.has_explicit_metric_threshold(query)):
            
            enhanced['metric_filters'] = base_analysis['metric_filters'].copy()
            modifications.append("inherited_metric_filters_fallback")
        
        # If no metrics of interest specified but query seems to want specific metrics
        if not enhanced.get('metrics_of_interest') and self.extract_mentioned_metrics(query):
            enhanced['metrics_of_interest'] = self.extract_mentioned_metrics(query)
            modifications.append("extracted_mentioned_metrics")
        
        enhanced['modifications_made'] = modifications
        enhanced['context_inherited'] = True
        
        return enhanced

    def apply_simple_inheritance(self, base_analysis, query):
        """Simple rule-based inheritance when LLM fails"""
        
        enhanced = base_analysis.copy()
        enhanced['context_inherited'] = True
        enhanced['modifications_made'] = ["simple_inheritance_fallback"]
        enhanced['confidence'] = 0.6  # Lower confidence for fallback
        
        # Extract any new metrics mentioned in the query
        mentioned_metrics = self.extract_mentioned_metrics(query)
        if mentioned_metrics:
            enhanced['metrics_of_interest'] = mentioned_metrics
            enhanced['modifications_made'].append("updated_metrics")
        
        # Adjust output format based on query patterns
        if any(word in query.lower() for word in ['details', 'break', 'daily', 'list', 'show all']):
            enhanced['output_format'] = 'detailed'
            enhanced['modifications_made'].append("changed_to_detailed")
        
        return enhanced

    def extract_mentioned_metrics(self, query):
        """Extract metric names mentioned in the query"""
        metric_mapping = {
            'oee': 'OEE',
            'quality': 'QualityRate',
            'quality rate': 'QualityRate', 
            'production': 'Production',
            'target': 'Target',
            'rejection': 'Rejection',
            'downtime': 'Downtime',
            'runtime': 'Run Time',
            'run time': 'Run Time',
            'availability': 'Availibility',
            'performance': 'Performance Efficiency',
            'efficiency': 'Performance Efficiency'
        }
        
        mentioned = []
        query_lower = query.lower()
        
        for keyword, metric_name in metric_mapping.items():
            if keyword in query_lower:
                mentioned.append(metric_name)
        
        return list(set(mentioned))  # Remove duplicates

    def has_explicit_time_reference(self, query):
        """Check if query explicitly mentions time periods"""
        time_keywords = [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'q1', 'q2', 'q3', 'q4', '2024', '2023', 'monday', 'tuesday', 
            'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'week', 'month', 'year', 'quarter'
        ]
        return any(keyword in query.lower() for keyword in time_keywords)

    def has_explicit_metric_threshold(self, query):
        """Check if query mentions specific metric thresholds"""
        threshold_patterns = ['>', '<', '>=', '<=', '==', 'greater', 'less', 'above', 'below', 'exactly']
        return any(pattern in query.lower() for pattern in threshold_patterns)

    def _get_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis using rule-based approach"""
        return {
            "date_filters": {
                "specific_dates": [],
                "date_ranges": [],
                "day_patterns": [],
                "relative_periods": [],
                "quarters": [],
                "months": []
            },
            "metric_filters": {
                "thresholds": [],
                "comparisons": [],
                "top_n": [],
                "exact_values": []
            },
            "intent": "analyze",
            "output_format": "summary",
            "analysis_type": "single",
            "multi_period": False,
            "metrics_of_interest": [],
            "confidence": 0.3
        }

    def store_conversation_context(self, user_query, query_analysis, filtered_df, llm_response):
        """Store the conversation context for follow-up questions"""
        context_entry = {
            'timestamp': datetime.now(),
            'user_query': user_query,
            'query_analysis': query_analysis,
            'filtered_df_summary': {
                'row_count': len(filtered_df),
                'date_range': (filtered_df['Date'].min(), filtered_df['Date'].max()),
                'columns': list(filtered_df.columns)
            },
            'llm_response_summary': llm_response[:200] + "..." if len(llm_response) > 200 else llm_response
        }
        
        self.conversation_history.append(context_entry)
        
        # Keep only last 3-5 exchanges to avoid token limits
        if len(self.conversation_history) > 3:
            self.conversation_history.pop(0)
        
        # Update current session state
        self.current_session['last_filtered_df'] = filtered_df.copy()
        self.current_session['last_query_analysis'] = query_analysis.copy()
        self.current_session['active_time_filters'] = query_analysis.get('date_filters', {})
        self.current_session['active_metric_filters'] = query_analysis.get('metric_filters', {})

    def is_followup_question(self, query):
        """Detect if this is a follow-up question"""
        followup_indicators = [
            # Pronouns referring to previous context
            'that', 'those', 'it', 'them', 'this', 'these', 'for'
            # Time references to previous queries
            'same period', 'same time', 'previous', 'earlier', 'before',
            # Comparative references
            'compare', 'versus', 'vs', 'against', 'difference',
            # Incomplete queries (just metric names)
            'show production', 'what about', 'how about',
            # Direct references
            'more details', 'break it down', 'expand', 'elaborate'
        ]
        
        query_lower = query.lower()
        has_followup_indicators = any(indicator in query_lower for indicator in followup_indicators)
        
        # Also check if query is unusually short (likely incomplete)
        is_short_query = len(query.split()) <= 4
        
        # Check if no explicit time period is mentioned
        time_keywords = ['january', 'february', 'march', 'q1', 'q2', 'q3', 'q4', '2024', 'monday', 'sunday']
        has_no_time_reference = not any(time_word in query_lower for time_word in time_keywords)
        
        return (has_followup_indicators or (is_short_query and has_no_time_reference)) and len(self.conversation_history) > 0

    def apply_smart_filters(self, query_analysis: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters based on query analysis using calculated metrics"""
        print(f"🔍 Applying smart filters based on LLM analysis...")
        print(f"📊 Starting with {len(self.df)} total records")
        
        filtered_df = self.df.copy()
        
        # Apply date filters first
        filtered_df = self.apply_date_filters(filtered_df, query_analysis["date_filters"])
        print(f"📅 After date filtering: {len(filtered_df)} records")
        
        # Apply metric filters using calculated values
        filtered_df = self.apply_metric_filters(filtered_df, query_analysis["metric_filters"])
        print(f"🎯 After metric filtering: {len(filtered_df)} records")
        
        return filtered_df

    def apply_date_filters(self, df: pd.DataFrame, date_filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply date-based filters to the DataFrame"""
        filtered_df = df.copy()
        
        # Apply specific dates
        if date_filters.get("specific_dates"):
            specific_dates = [pd.to_datetime(date) for date in date_filters["specific_dates"]]
            filtered_df = filtered_df[filtered_df['Date'].isin(specific_dates)]
        
        # Apply date ranges
        if date_filters.get("date_ranges"):
            range_dfs = []
            for date_range in date_filters["date_ranges"]:
                start_date = pd.to_datetime(date_range["start"])
                end_date = pd.to_datetime(date_range["end"])
                range_df = filtered_df[
                    (filtered_df['Date'] >= start_date) & 
                    (filtered_df['Date'] <= end_date)
                ]
                range_dfs.append(range_df)
            if range_dfs:
                filtered_df = pd.concat(range_dfs).drop_duplicates()
        
        # Apply day patterns
        if date_filters.get("day_patterns"):
            day_mapping = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            
            pattern_dfs = []
            for pattern in date_filters["day_patterns"]:
                if pattern.lower() in day_mapping:
                    day_num = day_mapping[pattern.lower()]
                    pattern_df = filtered_df[filtered_df['Date'].dt.dayofweek == day_num]
                    pattern_dfs.append(pattern_df)
                elif pattern.lower() == 'weekend':
                    weekend_df = filtered_df[filtered_df['Date'].dt.dayofweek.isin([5, 6])]
                    pattern_dfs.append(weekend_df)
                elif pattern.lower() == 'weekday':
                    weekday_df = filtered_df[filtered_df['Date'].dt.dayofweek.isin([0, 1, 2, 3, 4])]
                    pattern_dfs.append(weekday_df)
            
            if pattern_dfs:
                filtered_df = pd.concat(pattern_dfs).drop_duplicates()
        
        # Apply months
        if date_filters.get("months"):
            month_dfs = []
            for month_str in date_filters["months"]:
                if "_" in month_str:
                    month_name, year_str = month_str.split("_")
                    year = int(year_str)
                    
                    month_mapping = {
                        'january': 1, 'february': 2, 'march': 3, 'april': 4,
                        'may': 5, 'june': 6, 'july': 7, 'august': 8,
                        'september': 9, 'october': 10, 'november': 11, 'december': 12
                    }
                    
                    if month_name.lower() in month_mapping:
                        month_num = month_mapping[month_name.lower()]
                        month_df = filtered_df[
                            (filtered_df['Date'].dt.year == year) &
                            (filtered_df['Date'].dt.month == month_num)
                        ]
                        month_dfs.append(month_df)
            
            if month_dfs:
                filtered_df = pd.concat(month_dfs).drop_duplicates()
        
        # Apply quarters
        if date_filters.get("quarters"):
            quarter_dfs = []
            for quarter in date_filters["quarters"]:
                if "_" in quarter:
                    q_part, year_part = quarter.split("_")
                    year = int(year_part)
                    quarter_num = int(q_part[1])  # Extract number from "Q1"
                    
                    quarter_df = filtered_df[
                        (filtered_df['Date'].dt.year == year) &
                        (filtered_df['Date'].dt.quarter == quarter_num)
                    ]
                    quarter_dfs.append(quarter_df)
            
            if quarter_dfs:
                filtered_df = pd.concat(quarter_dfs).drop_duplicates()
        
        return filtered_df

    def apply_metric_filters(self, df: pd.DataFrame, metric_filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply metric-based filters using CALCULATED values"""
        filtered_df = df.copy()
        
        # Metrics that are calculated (use calculated columns)
        calculated_metrics = {
            'OEE': 'Calculated_OEE',
            'Performance Efficiency': 'Calculated_Performance_Efficiency', 
            'QualityRate': 'Calculated_QualityRate',
            'Availability': 'Calculated_Availability'
        }
        
        # Apply threshold filters
        if metric_filters.get("thresholds"):
            for threshold in metric_filters["thresholds"]:
                metric = threshold["metric"]
                operator = threshold["operator"]
                value = threshold["value"]
                
                print(f"🔍 Applying filter: {metric} {operator} {value}")
                
                # Use calculated column if available, otherwise use original column
                column_to_use = calculated_metrics.get(metric, metric)
                
                if column_to_use in filtered_df.columns:
                    original_count = len(filtered_df)
                    
                    if operator == ">":
                        filtered_df = filtered_df[filtered_df[column_to_use] > value]
                    elif operator == "<":
                        filtered_df = filtered_df[filtered_df[column_to_use] < value]
                    elif operator == ">=":
                        filtered_df = filtered_df[filtered_df[column_to_use] >= value]
                    elif operator == "<=":
                        filtered_df = filtered_df[filtered_df[column_to_use] <= value]
                    elif operator == "==":
                        filtered_df = filtered_df[abs(filtered_df[column_to_use] - value) < 0.0001]
                    
                    print(f"✅ Filtered from {original_count} to {len(filtered_df)} records using {column_to_use}")
        
        # Apply exact value filters
        if metric_filters.get("exact_values"):
            for exact_filter in metric_filters["exact_values"]:
                metric = exact_filter["metric"]
                value = exact_filter["value"]
                
                column_to_use = calculated_metrics.get(metric, metric)
                if column_to_use in filtered_df.columns:
                    filtered_df = filtered_df[abs(filtered_df[column_to_use] - value) < 0.0001]
        
        # Apply top N filters
        if metric_filters.get("top_n"):
            for top_filter in metric_filters["top_n"]:
                n = top_filter["n"]
                metric = top_filter["metric"]
                direction = top_filter.get("direction", "highest")
                
                column_to_use = calculated_metrics.get(metric, metric)
                if column_to_use in filtered_df.columns:
                    ascending = direction == "lowest"
                    filtered_df = filtered_df.nlargest(n, column_to_use) if not ascending else filtered_df.nsmallest(n, column_to_use)
        
        # Apply comparison filters (max/min)
        if metric_filters.get("comparisons"):
            for comparison in metric_filters["comparisons"]:
                comp_type = comparison.get("type")
                if not comp_type:
                    print("⚠️ Skipping comparison without 'type' key:", comparison)
                    continue

                metric = comparison["metric"]
                
                column_to_use = calculated_metrics.get(metric, metric)
                if column_to_use in filtered_df.columns:
                    if comp_type == "maximum":
                        max_value = filtered_df[column_to_use].max()
                        filtered_df = filtered_df[filtered_df[column_to_use] == max_value]
                    elif comp_type == "minimum":
                        min_value = filtered_df[column_to_use].min()
                        filtered_df = filtered_df[filtered_df[column_to_use] == min_value]
        
        return filtered_df

    def call_production_llm(self, context_text: str, query: str, period_averages=None):
        """Optimized LLM call for production data analysis"""
        
        headers = {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        # Production-specific system prompt
        system_prompt = (
            "You are a manufacturing data analysis expert. The user will ask about one or more metrics based on production logs.\n\n"
        
            "IMPORTANT FOR COUNTING: The data provided to you is already pre-filtered. If asked to count records, "
            "simply count what's provided - do not re-filter or re-analyze.\n\n"
    
            "BASE METRICS (available for each day):\n"
            "- Target: Daily production goal (units)\n"
            "- Production: Number of good units produced\n"
            "- Rejection: Number of defective units\n"
            "- Downtime: Equipment downtime (in hours)\n"
            "- Run Time: Machine operation time (in hours)\n\n"
            
            "DERIVED DAILY METRICS (computed per day):\n"
            "- Performance_Efficiency = (Production + Rejection) / Target\n"
            "- Quality_Rate = Production / (Production + Rejection)\n"
            "- Availability = Run Time / (Run Time + Downtime)\n"
            "- OEE = Availability * Performance_Efficiency * Quality_Rate\n\n"
            
            "PERIOD METRIC RULES:\n"
            "- Period averages must be calculated using totals across the selected time period:\n"
            "  - Total Production = sum of daily Production\n"
            "  - Total Rejection = sum of daily Rejection\n"
            "  - Total Target = sum of daily Target\n"
            "  - Total Run Time = sum of daily Run Time\n"
            "  - Total Downtime = sum of daily Downtime\n\n"
            
            "Then:\n"
            "- Period Performance_Efficiency = (Total Production + Total Rejection) / Total Target\n"
            "- Period Quality_Rate = Total Production / (Total Production + Total Rejection)\n"
            "- Period Availability = Total Run Time / (Total Run Time + Total Downtime)\n"
            "- Period OEE = Availability * Performance_Efficiency * Quality_Rate\n\n"
            
            "ALWAYS:\n"
            "1. Use period formulas for any month/year range queries.\n"
            "2. Use daily formulas when the user asks for per-day values or trends.\n"
            "3. Show the applied formula and exact values used in calculations.\n"
            "4. Be precise: round rate metrics to 4 decimal places.\n"
            "5. Sort daily results chronologically.\n\n"
            
            "RESPONSE FORMAT:\n"
            "- Summarize totals and derived averages first.\n"
            "- Include formulas clearly.\n"
            "- Include per-day breakdowns only when the user explicitly asks for them (e.g., 'daily data', 'show all days', etc).\n\n"
            
            "DISPLAY RULES:\n"
            "- Be concise; prioritize summaries and insights over raw data.\n"
            "- Only list daily records when asked.\n"
            "- If comparing across time periods (months or years), highlight trends like improvement or decline in key metrics (e.g., OEE, Production).\n"
            "- Only show dates where the value matches the exact condition asked (e.g., rejection == 0). No approximations.\n"
            "- Never list entries that don't match the user's filter condition. Exclude them from output entirely.\n"
            "- When analyzing one metric (e.g., Availability), focus only on that and do not include unrelated metrics unless asked.\n\n"
            
            "You are expected to explain the insights like a senior analyst presenting to management.\n"
            "Avoid vague statements and provide specific numeric evidence for conclusions.\n\n"
        )
        if period_averages:
            system_prompt += (
                "PERIOD AVERAGES (calculated using formulas on totals):\n"
                f"- Period Performance Efficiency: {period_averages.get('Performance_Efficiency', 0):.4f}\n"
                f"- Period Quality Rate: {period_averages.get('Quality_Rate', 0):.4f}\n"
                f"- Period Availability: {period_averages.get('Availability', 0):.4f}\n"
                f"- Period OEE: {period_averages.get('OEE', 0):.4f}\n\n"
                
                "CALCULATION BREAKDOWN:\n"
                f"- Total Production: {period_averages['Totals']['Production']:.0f} units\n"
                f"- Total Rejection: {period_averages['Totals']['Rejection']:.0f} units\n"
                f"- Total Target: {period_averages['Totals']['Target']:.0f} units\n"
                f"- Total Runtime: {period_averages['Totals']['Runtime']:.0f} hours\n"
                f"- Total Downtime: {period_averages['Totals']['Downtime']:.0f} hours\n"
                f"- Number of Days: {period_averages['Totals']['Days']}\n\n"
                
                "IMPORTANT: ALWAYS USE THESE VALUES TO CALCULATE ANYTHING FURTHER\n"
                "Show the calculation\n\n"
            )
            
        system_prompt += (
            "ANALYSIS RULES:\n"
            "1. Use the provided PERIOD values calculated from totals\n"
            "2. For daily analysis: Use the individual day calculations\n"
            "3. Always show calculation formulas when requested\n"
            "4. Sort results chronologically by date\n"
            "5. Be precise with decimal places (4 decimal places for rates)\n"
            "6. Do NOT fabricate or assume dates. Only use the dates explicitly found in the metadata.\n"
            "7. Only include dates where the condition is exactly met.\n"
                "   - Example: If asked for 'exactly 2 hours of downtime', do NOT include 0, 1, or 3 hours.\n"
                "   - Do not explain or show entries that don't meet the condition. Exclude them entirely.\n"
                "   - Never infer or assume values — only use explicitly matched records.\n"
            "8. When asked for records with conditions (e.g., Rejection = 0), directly filter using metadata, not LLM assumptions.\n\n"

            "RESPONSE FORMAT:\n"
            "- For averages: Show formula calculation with totals\n"
            "- For individual days: List chronologically with values\n"
            "- Always be clear about what type of calculation you're showing\n"
        )

        payload = {
            "model": self.model,
            "max_tokens": 3000,
            "temperature": 0.1,  # Low temperature for consistent calculations
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Production Data:\n{context_text}\n\nAnalyze: {query}"}
            ]
        }

        try:
            start_time = time.time()
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                print(f"⚡ Response time: {response_time:.2f}s")
                return result
            else:
                return f"❌ LLM Error ({response.status_code}): {response.text[:200]}..."
                
        except requests.exceptions.Timeout:
            return "⏰ Request timed out. Try a simpler query."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def call_production_llm_multi_period(self, context_text: str, query: str, individual_period_averages=None):
        """LLM call for multi-period analysis - shows each period separately"""
        
        headers = {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        system_prompt = (
            "You are a manufacturing data analysis expert. The user is asking about multiple time periods and wants to see each period's results SEPARATELY.\n\n"
            
            "CRITICAL INSTRUCTION: Use the provided INDIVIDUAL PERIOD AVERAGES for each time period. \n"
            "DO NOT combine or recalculate them. Report each period separately and compare if relevant.\n\n"

            "CRITICAL INSTRUCTION FOR COUNTING QUERIES:\n"
            "When user asks 'how many', 'number of', 'count', etc.:\n"
            "1. Count EVERY single record in the provided data\n"
            "2. The data is already pre-filtered - do NOT re-filter it\n"
            "3. Simply count the number of data rows (excluding headers)\n"
            "4. Do NOT skip any records in your count\n"
            "5. Your count MUST match the total number of records provided\n\n"
            
            "BASE METRICS (available for each day):\n"
            "- Target: Daily production goal (units)\n"
            "- Production: Number of good units produced\n"
            "- Rejection: Number of defective units\n"
            "- Downtime: Equipment downtime (in hours)\n"
            "- Run Time: Machine operation time (in hours)\n\n"
            
            "DERIVED METRICS (calculated per period):\n"
            "- Performance_Efficiency = (Total Production + Total Rejection) / Total Target\n"
            "- Quality_Rate = Total Production / (Total Production + Total Rejection)\n"
            "- Availability = Total Runtime / (Total Runtime + Total Downtime)\n"
            "- OEE = Availability * Performance_Efficiency * Quality_Rate\n\n"
        )
        
        if individual_period_averages:
            system_prompt += "INDIVIDUAL PERIOD AVERAGES (USE THESE EXACT VALUES):\n"
            system_prompt += "=" * 80 + "\n"
            
            for period_key, period_data in individual_period_averages.items():
                period_name = period_data['name']
                averages = period_data['averages']
                
                system_prompt += f"📅 {period_name.upper()}:\n"
                system_prompt += f"  ✅ Quality Rate: {averages['Quality_Rate']:.4f} ({averages['Quality_Rate']*100:.2f}%)\n"
                system_prompt += f"  ✅ Performance Efficiency: {averages['Performance_Efficiency']:.4f} ({averages['Performance_Efficiency']*100:.2f}%)\n"
                system_prompt += f"  ✅ Availability: {averages['Availability']:.4f} ({averages['Availability']*100:.2f}%)\n"
                system_prompt += f"  ✅ OEE: {averages['OEE']:.4f} ({averages['OEE']*100:.2f}%)\n"
                system_prompt += f"  📊 Based on {averages['Totals']['Days']} days of data\n"
                system_prompt += f"  📈 Totals: {averages['Totals']['Production']:.0f} production, {averages['Totals']['Rejection']:.0f} rejection, {averages['Totals']['Target']:.0f} target\n\n"
            
            system_prompt += "=" * 80 + "\n"
            system_prompt += "MANDATORY: Report each period's values separately using the ✅ values above.\n"
            system_prompt += "Show comparisons and trends between periods when relevant.\n"
            system_prompt += "=" * 80 + "\n\n"
        
        system_prompt += (
            "RESPONSE FORMAT:\n"
            "1. Answer each time period separately first\n"
            "2. Use format: '[Period Name]: [Metric] is [value] ([percentage]%)'\n"
            "3. After individual results, provide comparison insights\n"
            "4. Highlight trends (improvement, decline, consistency)\n"
            "5. Be specific with numbers and percentages\n\n"
            
            "EXAMPLE RESPONSE STRUCTURE:\n"
            "**Individual Period Results:**\n"
            "• January 2024: Quality Rate is 0.9850 (98.50%)\n"
            "• February 2024: Quality Rate is 0.9920 (99.20%)\n\n"
            "**Comparison & Insights:**\n"
            "February showed a 0.70 percentage point improvement over January.\n"
            "This represents a 0.71% relative improvement in quality performance.\n"
            "The trend indicates improving quality control processes.\n\n"
            
            "ANALYSIS RULES:\n"
            "- Always show both periods individually first\n"
            "- Calculate percentage point differences (absolute difference)\n"
            "- Calculate relative percentage changes when meaningful\n"
            "- Identify which period performed better and by how much\n"
            "- Comment on trends and patterns\n"
            "- Be precise with decimal places (4 decimal places for rates)\n"
            "- Don't make assumptions about causes, stick to the data\n"
        )

        payload = {
            "model": self.model,
            "max_tokens": 3000,
            "temperature": 0.0,  # Consistent results
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Production Data:\n{context_text}\n\nAnalyze: {query}"}
            ]
        }

        try:
            start_time = time.time()
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                print(f"⚡ Response time: {response_time:.2f}s")
                return result
            else:
                return f"❌ LLM Error ({response.status_code}): {response.text[:200]}..."
                
        except requests.exceptions.Timeout:
            return "⏰ Request timed out. Try a simpler query."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def format_production_table(self, records):
        """Format production data with calculated metrics"""
        lines = []
        for r in records:
            line = (f"Date: {r.get('Date', 'N/A')}, "
                    f"Production: {r.get('Production', 'N/A')}, "
                    f"Rejection: {r.get('Rejection', 'N/A')}, "
                    f"Target: {r.get('Target', 'N/A')}, "
                    f"Run Time: {r.get('Run Time', 'N/A')}, "
                    f"Downtime: {r.get('Downtime', 'N/A')}")
            
            # Add calculated metrics
            line += (f", Performance_Efficiency: {r.get('Performance Efficiency', 0):.4f}, "
                    f"Quality_Rate: {r.get('QualityRate', 0):.4f}, "
                    f"Availability: {r.get('Availibility', 0):.4f}, "
                    f"OEE: {r.get('OEE', 0):.4f}")
            
            lines.append(line)
        return "\n".join(lines)

    def calculate_period_averages(self, records):
        """Calculate period averages using formulas on totals"""
        try:
            # Extract totals from all records
            total_production = sum(float(r.get('Production', 0)) for r in records if r.get('Production'))
            total_rejection = sum(float(r.get('Rejection', 0)) for r in records if r.get('Rejection'))
            total_target = sum(float(r.get('Target', 0)) for r in records if r.get('Target'))
            total_runtime = sum(float(r.get('Run Time', 0)) for r in records if r.get('Run Time'))
            total_downtime = sum(float(r.get('Downtime', 0)) for r in records if r.get('Downtime'))
            
            # Calculate period averages using formulas
            period_averages = {}
            
            if total_target > 0:
                period_averages['Performance_Efficiency'] = (total_production + total_rejection) / total_target
            else:
                period_averages['Performance_Efficiency'] = 0
                
            total_produced = total_production + total_rejection
            if total_produced > 0:
                period_averages['Quality_Rate'] = total_production / total_produced
            else:
                period_averages['Quality_Rate'] = 0
                
            # Availability = Total Runtime / (Total Runtime + Total Downtime)
            total_time = total_runtime + total_downtime
            if total_time > 0:
                period_averages['Availability'] = total_runtime / total_time
            else:
                period_averages['Availability'] = 0
                
            # OEE = Availability * Performance Efficiency * Quality Rate
            period_averages['OEE'] = (period_averages['Availability'] * 
                                    period_averages['Performance_Efficiency'] * 
                                    period_averages['Quality_Rate'])
            
            # Add totals for reference
            period_averages['Totals'] = {
                'Production': total_production,
                'Rejection': total_rejection,
                'Target': total_target,
                'Runtime': total_runtime,
                'Downtime': total_downtime,
                'Days': len(records)
            }
            
            return period_averages
            
        except Exception as e:
            print(f"❌ Error calculating period averages: {e}")
            return {}

    def is_average_query(self, query: str) -> bool:
        """Check if query is asking for period averages"""
        average_keywords = [
            "average", "mean", "avg", "overall", "total", "period",
            "monthly", "quarterly", "yearly", "for the month", 
            "for the quarter", "for the year", "calculate average", "combined", "for"
        ]
        return any(keyword in query.lower() for keyword in average_keywords)

    def ask_question(self, query: str):
        """Enhanced query processing with conversation history and LLM-based pre-filtering"""
        start_time = time.time()
        
        if not query.strip():
            return "Please provide a valid question about production data."

        print(f"🧠 Analyzing query with LLM to extract filtering criteria...")
        
        # Step 1: Enhanced query analysis (now context-aware)
        query_analysis = self.analyze_query(query, available_columns=list(self.df.columns))
        
        print(f"🎯 Query Analysis Results:")
        print(f"   Intent: {query_analysis.get('intent', 'unknown')}")
        print(f"   Output Format: {query_analysis.get('output_format', 'summary')}")
        print(f"   Analysis Type: {query_analysis.get('analysis_type', 'single')}")
        print(f"   Multi Period: {query_analysis.get('multi_period', False)}")
        print(f"   Confidence: {query_analysis.get('confidence', 0.0):.2f}")
        print(f"   Metrics of Interest: {query_analysis.get('metrics_of_interest', [])}")
        print(f"   Context Inherited: {query_analysis.get('context_inherited', False)}")
        if query_analysis.get('context_inherited'):
            print(f"   Modifications Made: {query_analysis.get('modifications_made', [])}")
        
        # Step 2: Smart filtering with context awareness
        # For follow-ups, potentially reuse last filtered data if filters haven't changed
        if (self.is_followup_question(query) and 
            self.current_session['last_filtered_df'] is not None and
            not self.has_new_filters(query_analysis) and
            query_analysis.get('context_inherited', False)):
            
            print("🔄 Reusing previous filter results for follow-up question...")
            filtered_df = self.current_session['last_filtered_df'].copy()
            
            # Apply any new filters that might have been added in the follow-up
            # if self.has_additional_filters(query_analysis):
            #     print("🎯 Applying additional filters from follow-up...")
            #     filtered_df = self.apply_smart_filters_incremental(filtered_df, query_analysis)
            
        else:
            # Apply fresh filtering
            print("🔍 Applying fresh smart filters...")
            filtered_df = self.apply_smart_filters(query_analysis)
        
        print(f"📊 Filtering completed: {len(filtered_df)} relevant records")
        
        if filtered_df.empty:
            # Store empty result for context but inform user
            self.store_conversation_context(query, query_analysis, filtered_df, "No records found")
            return "❌ No records found matching your criteria."
        
        # Step 3: Convert to records format for LLM processing
        filtered_records = filtered_df.to_dict('records')
        
        # Convert date objects to strings for consistency
        for record in filtered_records:
            if 'Date' in record and hasattr(record['Date'], 'strftime'):
                record['Date'] = record['Date'].strftime('%Y-%m-%d')
        
        # Step 4: Handle count queries quickly
        if query_analysis.get('intent') == 'count':
            count_result = f"There are {len(filtered_records)} records matching your criteria."
            self.store_conversation_context(query, query_analysis, filtered_df, count_result)
            return count_result
        
        # Step 5: Calculate period averages if needed
        period_averages = None
        if query_analysis.get('intent') in ['calculate', 'analyze'] or self.is_average_query(query):
            period_averages = self.calculate_period_averages(filtered_records)
        
        # Step 6: Generate enhanced query with context
        enhanced_query = self.build_enhanced_query_prompt(query, query_analysis)
        
        # Step 7: Determine which LLM method to use based on analysis type
        analysis_type = query_analysis.get('analysis_type', 'single')
        multi_period = query_analysis.get('multi_period', False)

        if multi_period and analysis_type == 'separate':
            # Multi-period separate analysis - need to calculate individual period averages
            print("🔄 Using multi-period separate analysis...")
            
            # Group filtered data by time periods for separate analysis
            individual_period_averages = self.calculate_individual_period_averages(
                filtered_records, query_analysis
            )
            
            # Call multi-period separate LLM
            context_text = self.format_production_table(filtered_records)
            result = self.call_production_llm_multi_period(
                context_text, enhanced_query, individual_period_averages
            )

        elif multi_period and analysis_type == 'combined':
            # Multi-period combined analysis - calculate overall period averages
            print("🔄 Using multi-period combined analysis...")
            period_averages = self.calculate_period_averages(filtered_records)
            context_text = self.format_production_table(filtered_records)
            result = self.call_production_llm(context_text, enhanced_query, period_averages)

        else:
            # Single period analysis (default)
            print("🔄 Using single period analysis...")
            context_text = self.format_production_table(filtered_records)
            result = self.call_production_llm(context_text, enhanced_query, period_averages)
        
        # Step 8: Store this conversation for future follow-ups
        self.store_conversation_context(query, query_analysis, filtered_df, result)
        
        total_time = time.time() - start_time
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        print(f"🚀 Efficiency gain: Processed only {len(filtered_records)}/{len(self.df)} records ({len(filtered_records)/len(self.df)*100:.1f}%)")
        
        return result

    def build_enhanced_query_prompt(self, original_query: str, query_analysis: Dict[str, Any]) -> str:
        """Build enhanced query prompt with analysis context"""
        
        enhanced_parts = [f"Original Query: {original_query}"]
        
        # Add query analysis context
        enhanced_parts.append("Query Analysis:")
        enhanced_parts.append(f"- Intent: {query_analysis.get('intent', 'analyze')}")
        enhanced_parts.append(f"- Preferred Output: {query_analysis.get('output_format', 'summary')}")
        enhanced_parts.append(f"- Key Metrics: {', '.join(query_analysis.get('metrics_of_interest', []))}")
        
        # Add conversation context if this is a follow-up
        if query_analysis.get('context_inherited', False):
            enhanced_parts.append("Conversation Context:")
            enhanced_parts.append("- This is a follow-up question to previous queries")
            enhanced_parts.append(f"- Context modifications: {', '.join(query_analysis.get('modifications_made', []))}")
            
            # Add recent conversation summary
            if self.conversation_history:
                last_exchange = self.conversation_history[-1]
                enhanced_parts.append(f"- Previous query: '{last_exchange['user_query']}'")
                enhanced_parts.append(f"- Previous intent: {last_exchange['query_analysis'].get('intent', 'unknown')}")
        
        enhanced_parts.append("\nBased on the analysis above, please provide a focused response.")
        
        return "\n".join(enhanced_parts)

    def has_new_filters(self, query_analysis: Dict[str, Any]) -> bool:
        """Check if the new query has different filters than the previous one"""
        if not self.current_session['last_query_analysis']:
            return True
        
        prev_analysis = self.current_session['last_query_analysis']
        
        # Compare date filters
        current_date_filters = query_analysis.get('date_filters', {})
        prev_date_filters = prev_analysis.get('date_filters', {})
        
        # Compare metric filters
        current_metric_filters = query_analysis.get('metric_filters', {})
        prev_metric_filters = prev_analysis.get('metric_filters', {})
        
        # Check if filters are different
        date_filters_changed = (
            current_date_filters.get('specific_dates', []) != prev_date_filters.get('specific_dates', []) or
            current_date_filters.get('date_ranges', []) != prev_date_filters.get('date_ranges', []) or
            current_date_filters.get('day_patterns', []) != prev_date_filters.get('day_patterns', []) or
            current_date_filters.get('quarters', []) != prev_date_filters.get('quarters', []) or
            current_date_filters.get('months', []) != prev_date_filters.get('months', [])
        )
        
        metric_filters_changed = (
            current_metric_filters.get('thresholds', []) != prev_metric_filters.get('thresholds', []) or
            current_metric_filters.get('comparisons', []) != prev_metric_filters.get('comparisons', []) or
            current_metric_filters.get('top_n', []) != prev_metric_filters.get('top_n', []) or
            current_metric_filters.get('exact_values', []) != prev_metric_filters.get('exact_values', [])
        )
        
        return date_filters_changed or metric_filters_changed

    def has_additional_filters(self, query_analysis: Dict[str, Any]) -> bool:
        """Check if the follow-up query has additional filters beyond the inherited ones"""
        if not query_analysis.get('context_inherited', False):
            return False
        
        # If the query analysis indicates new filters were added (not just inherited)
        modifications = query_analysis.get('modifications_made', [])
        additional_filter_modifications = [
            'added_new_thresholds', 'added_metric_filters', 'added_date_constraints',
            'extracted_mentioned_metrics', 'changed_to_detailed'
        ]
        
        return any(mod in additional_filter_modifications for mod in modifications)

    def apply_smart_filters_incremental(self, existing_df: pd.DataFrame, query_analysis: Dict[str, Any]) -> pd.DataFrame:
        """Apply additional filters to already filtered data"""
        print("🎯 Applying incremental filters to existing results...")
        
        # Only apply new metric filters that weren't inherited
        new_metric_filters = query_analysis.get('metric_filters', {})
        
        if any(new_metric_filters.values()):
            filtered_df = self.apply_metric_filters(existing_df, new_metric_filters)
            print(f"📊 After incremental filtering: {len(filtered_df)} records")
            return filtered_df
        
        return existing_df

    def calculate_individual_period_averages(self, filtered_records: List[Dict], query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate individual period averages for multi-period separate analysis"""
        individual_period_averages = {}
        
        # Extract unique time periods from the filtered data
        date_filters = query_analysis.get('date_filters', {})
        
        if date_filters.get('months'):
            # Group by months
            for i, month_str in enumerate(date_filters['months']):
                month_records = []
                if "_" in month_str:
                    month_name, year_str = month_str.split("_")
                    year = int(year_str)
                    month_mapping = {
                        'january': 1, 'february': 2, 'march': 3, 'april': 4,
                        'may': 5, 'june': 6, 'july': 7, 'august': 8,
                        'september': 9, 'october': 10, 'november': 11, 'december': 12
                    }
                    if month_name.lower() in month_mapping:
                        month_num = month_mapping[month_name.lower()]
                        month_records = [r for r in filtered_records 
                                    if pd.to_datetime(r['Date']).year == year and 
                                        pd.to_datetime(r['Date']).month == month_num]
                
                if month_records:
                    period_averages = self.calculate_period_averages(month_records)
                    individual_period_averages[f"period_{i}"] = {
                        'name': f"{month_name.title()} {year}",
                        'averages': period_averages
                    }
        
        elif date_filters.get('quarters'):
            # Group by quarters
            for i, quarter in enumerate(date_filters['quarters']):
                quarter_records = []
                if "_" in quarter:
                    q_part, year_part = quarter.split("_")
                    year = int(year_part)
                    quarter_num = int(q_part[1])
                    quarter_records = [r for r in filtered_records 
                                    if pd.to_datetime(r['Date']).year == year and 
                                        pd.to_datetime(r['Date']).quarter == quarter_num]
                
                if quarter_records:
                    period_averages = self.calculate_period_averages(quarter_records)
                    individual_period_averages[f"period_{i}"] = {
                        'name': f"Q{quarter_num} {year}",
                        'averages': period_averages
                    }
        
        return individual_period_averages

    def show_conversation_history(self):
        """Display recent conversation history"""
        if not self.conversation_history:
            print("📝 No conversation history available")
            return
        
        print("\n📝 Recent Conversation History:")
        print("=" * 50)
        
        for i, entry in enumerate(self.conversation_history[-3:], 1):  # Show last 3 exchanges
            print(f"\n{i}. Query: '{entry['user_query']}'")
            print(f"   Intent: {entry['query_analysis'].get('intent', 'unknown')}")
            print(f"   Records Found: {entry['filtered_df_summary']['row_count']}")
            print(f"   Date Range: {entry['filtered_df_summary'].get('date_range', 'N/A')}")
            print(f"   Response Preview: {entry['llm_response_summary']}")
        
        print("=" * 50)
        
        # Show current active context
        if self.current_session['last_query_analysis']:
            print("\n🎯 Current Active Context:")
            print(f"   Time Filters: {self.current_session['active_time_filters']}")
            print(f"   Metric Filters: {self.current_session['active_metric_filters']}")
            print(f"   Last Dataset Size: {len(self.current_session['last_filtered_df']) if self.current_session['last_filtered_df'] is not None else 0} records")


    def chat(self):
        """Enhanced interactive chat with smart filtering"""
        print("🏭 Enhanced Production Data Analysis Chatbot")
        print("=" * 60)
        print("🧠 Now with Smart LLM Query Analysis!")
        print(f"Model: {self.model}")
        print(f"Data Source: {self.csv_file_path}")
        print()
        print("📊 Enhanced capabilities:")
        print("• 'Show quality rate for all Sundays in Q1 2024'")
        print("• 'List top 5 days with highest OEE'")
        print("• 'Find all weekends with zero downtime'")
        print("• 'Compare production on Mondays vs Fridays'")
        print("• 'Tell the number of dates where oee > 0.7'")
        print("• 'stats' - Quick statistics overview")
        print("• 'quit' - Exit")
        print("=" * 60)
        
        while True:
            try:
                query = input(f"\n🔍 You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                answer = self.ask_question(query)
                print(f"🤖 Smart Analysis:\n{answer}")
                print("="*80)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        chatbot = ProductionDataChatbot("data1.csv")
        chatbot.chat()
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")
        print("Please check your CSV file path and API keys.")