*Production Data Analysis Chatbot*

An intelligent chatbot for analyzing manufacturing production data with natural language queries. Powered by LLM-based query analysis and smart filtering capabilities.


**Features**
- Intelligent Query Processing
    - Natural Language Understanding: Ask questions in plain English about your production data
    - Context-Aware Conversations: Follow-up questions maintain context from previous queries
    - Smart Filtering: Automatically extracts filtering criteria from natural language queries
    - Multi-Period Analysis: Compare data across different time periods (months, quarters, days)

- Manufacturing Metrics
    - OEE (Overall Equipment Effectiveness): Comprehensive efficiency calculation
    - Quality Rate: Production quality analysis with rejection tracking
    - Availability: Equipment uptime and downtime analysis
    - Performance Efficiency: Production vs target achievement
    - Time-Based Analysis: Day patterns, date ranges, and period comparisons

- Advanced Capabilities
    - Follow-up Context: Remembers previous questions for natural conversation flow
    - Pre-calculated Metrics: Optimized performance with pre-computed derived metrics
    - Flexible Time Filtering: Supports dates, quarters, months, weekdays, and custom ranges
    - Multi-Analysis Types: Single period, combined periods, or separate period analysis



**Usage**
*Example Queries*
    - "Show quality rate for all Sundays in Q1 2024"
    -  "List top 5 days with highest OEE"
    - "Find all weekends with zero downtime"
    - "Compare production on Mondays vs Fridays"
    - "How many dates had OEE > 0.7?"
    - "What was the average OEE for January 2024?"
    - "Show me production trends for February"

*Follow-up Questions*
The chatbot maintains conversation context:

    User: "Show OEE for January 2024"
    Bot: [Shows January OEE analysis]

    User: "What about February?"
    Bot: [Automatically shows February OEE, maintaining same analysis approach]

    User: "Compare those two months"
    Bot: [Compares January vs February OEE with insights]



**Data Format**

*Column*         *Description*                   *Type*
Date             Production Date                 Date (YYYY-MM-DD)
Target           Daily Production Goal           Numeric
Production       Good Units Produced             Numeric             
Rejection        Defective Units                 Numeric
Run Time         Machine Operation Hours         Numeric    
Downtime         Equipment Downtime Hours        Numeric     

- Optional Pre-calculated Columns
    The system can work with existing calculated metrics, but will auto-calculate if missing:
    - QualityRate
    - Availability (note: uses "Availibility" to match existing data)
    - Performance Efficiency
    - OEE

**Architecture**

*Core Components*
- Query Analysis Engine
    - LLM-Powered Analysis: Uses Gemini 2.5 Flash Lite for query understanding
    - Structured Extraction: Converts natural language to structured filtering criteria
    - Context Inheritance: Maintains conversation state for follow-up questions

- Smart Filtering System
    - Date Filtering: Handles specific dates, ranges, quarters, months, and day patterns
    - Metric Filtering: Applies thresholds, comparisons, top-N selections, and exact matches
    - Calculated Metrics: Uses pre-computed derived metrics for consistent filtering

- Conversation Management
    - Context Storage: Maintains conversation history for natural follow-up questions
    - Session State: Tracks active filters and previous results
    - Intelligent Inheritance: Automatically inherits relevant context in follow-ups



**Calculated Metrics Formulas**

- Performance Efficiency
    PE = (Production + Rejection) / Target

- Quality Rate  
    QR = Production / (Production + Rejection)

- Availability
    A = Run Time / (Run Time + Downtime)

- Overall Equipment Effectiveness (OEE)
    OEE = Availability × Performance_Efficiency × Quality_Rate



**Configuration**

- Environment Variables
    - OPENROUTER_API_KEY: Your OpenRouter API key for LLM functionality

- Model Configuration
    - Default model: google/gemini-2.5-flash-lite
    - Configurable in the ProductionDataChatbot.__init__() method

- Performance Settings
    - Context retention: 3-5 recent exchanges
    - Response timeout: 30 seconds
    - Temperature: 0.1 (for consistent analysis)
