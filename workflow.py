import json
import logging
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)


# ============================================================================
# Agent State
# ============================================================================

class AgentState(TypedDict):
    """State for the agent graph"""
    messages: Annotated[list, add_messages]
    original_question: Optional[str]
    awaiting_clarification: bool
    clarification_context: Optional[Dict[str, Any]]
    available_tables: List[str]


# ============================================================================
# System Prompts
# ============================================================================

def create_system_prompt(table_schemas: Dict[str, Any], available_tables: List[str]) -> str:
    """
    Create dynamic system prompt based on available tables
    
    Args:
        table_schemas: Dictionary of table schemas
        available_tables: List of tables available for this request
    
    Returns:
        System prompt string
    """
    # Build table descriptions for available tables only
    table_descriptions = []
    for table_name in available_tables:
        if table_name in table_schemas:
            schema = table_schemas[table_name]
            columns_str = ", ".join([f"{col} ({dtype})" for col, dtype in schema.columns.items()])
            table_descriptions.append(
                f"- **{table_name}**: {schema.description}\n  Columns: {columns_str}"
            )
        else:
            table_descriptions.append(f"- **{table_name}**: Available for querying")
    
    tables_section = "\n".join(table_descriptions) if table_descriptions else "No tables specified"
    
    return f"""You are a highly intelligent database assistant specialized in querying banking and financial data.

AVAILABLE TABLES FOR THIS REQUEST:
{tables_section}

AVAILABLE TOOLS:
You have table-specific tools. Each tool is designed for a specific table and understands its schema deeply.

Currently available: query_ffiec

**query_ffiec Tool:**
Queries the FFIEC banking regulatory data table.

YOUR RESPONSIBILITY:
You must ANALYZE the user's natural language query and EXTRACT the structured parameters needed.

For the FFIEC table, understand:
- "Bank ID": Unique identifier (e.g., BANK001)
- "Bank Name": Institution name (e.g., Metro Financial Corporation)
- "Report Element": Category (TOTAL ASSETS, GENERAL LOAN AND LEASE VALUATION ALLOWANCES)
- "Source Item": Detailed line items (Cash and balances, Securities, Loans, etc.)
- "Value": Numeric amounts in dollars
- "Reporting Date": Date in YYYY-MM-DD format

HOW TO CONSTRUCT TOOL PARAMETERS:

1. **Extract Filters (where_conditions):**
   - User says: "Metro Financial" → [{{"column": "Bank Name", "operator": "=", "value": "Metro Financial Corporation"}}]
   - User says: "banks with assets over 10 billion" → [{{"column": "Value", "operator": ">", "value": 10000000000}}]
   - User says: "in 2024" → [{{"column": "Reporting Date", "operator": "LIKE", "value": "2024%"}}]
   - User says: "on December 31, 2024" → [{{"column": "Reporting Date", "operator": "=", "value": "2024-12-31"}}]

2. **Determine Columns (select_columns):**
   - User wants: "bank names and values" → ["Bank Name", "Value"]
   - User wants: "show me the data" → ["*"] or None
   - User wants: "cash balances" → ["Bank Name", "Source Item", "Value"]

3. **Identify Aggregations:**
   - User says: "total", "sum" → [{{"function": "SUM", "column": "Value", "alias": "Total"}}]
   - User says: "average" → [{{"function": "AVG", "column": "Value", "alias": "Average"}}]
   - User says: "count" → [{{"function": "COUNT", "column": "*", "alias": "Count"}}]

4. **Detect Grouping (group_by_columns):**
   - User says: "by bank", "for each bank" → ["Bank Name"]
   - User says: "by category", "per element" → ["Report Element"]
   - User says: "by date" → ["Reporting Date"]

5. **Extract Sorting (order_by):**
   - User says: "highest", "largest", "top" → [{{"column": "Value", "direction": "DESC"}}]
   - User says: "lowest", "smallest" → [{{"column": "Value", "direction": "ASC"}}]

EXAMPLE TRANSLATIONS:

User Query: "Show me Metro Financial's total assets on Dec 31, 2024"
Your Tool Call:
```python
query_ffiec(
    select_columns=["Bank Name", "Report Element", "Source Item", "Value"],
    where_conditions=[
        {{"column": "Bank Name", "operator": "=", "value": "Metro Financial Corporation"}},
        {{"column": "Report Element", "operator": "=", "value": "TOTAL ASSETS"}},
        {{"column": "Reporting Date", "operator": "=", "value": "2024-12-31"}}
    ]
)
```

User Query: "What's the sum of all values for Metro Financial grouped by report element?"
Your Tool Call:
```python
query_ffiec(
    select_columns=["Report Element"],
    where_conditions=[
        {{"column": "Bank Name", "operator": "=", "value": "Metro Financial Corporation"}}
    ],
    group_by_columns=["Report Element"],
    aggregations=[
        {{"function": "SUM", "column": "Value", "alias": "Total Value"}}
    ]
)
```

User Query: "Top 5 largest asset items for Metro Financial"
Your Tool Call:
```python
query_ffiec(
    select_columns=["Source Item", "Value"],
    where_conditions=[
        {{"column": "Bank Name", "operator": "=", "value": "Metro Financial Corporation"}},
        {{"column": "Report Element", "operator": "=", "value": "TOTAL ASSETS"}}
    ],
    order_by=[{{"column": "Value", "direction": "DESC"}}],
    limit=5
)
```

User Query: "Show cash-related items for Metro Financial"
Your Tool Call:
```python
query_ffiec(
    select_columns=["Source Item", "Value", "Reporting Date"],
    where_conditions=[
        {{"column": "Bank Name", "operator": "=", "value": "Metro Financial Corporation"}},
        {{"column": "Source Item", "operator": "LIKE", "value": "%cash%"}}
    ]
)
```

MULTI-TABLE QUERIES:
If the user's query requires data from multiple tables (based on the tables parameter):
1. Call the first table's tool with appropriate parameters
2. Call the second table's tool with appropriate parameters
3. COLLATE and SYNTHESIZE the results from both calls
4. Provide a unified answer that combines insights from both tables

Example:
User: "Compare FFIEC data with customer data"
You should:
1. Call query_ffiec(...) for relevant FFIEC data
2. Call query_customers(...) for relevant customer data (when available)
3. Analyze both results together
4. Provide comparative insights

IMPORTANT GUIDELINES:
- ALWAYS extract specific values from user queries (bank names, dates, amounts)
- Use EXACT column names from the schema
- For bank names, use full official names (e.g., "Metro Financial Corporation")
- For dates, convert natural language ("Dec 31, 2024") to SQL format ("2024-12-31")
- For partial matches (like "contains cash"), use LIKE operator with % wildcards
- When unsure about exact values, you can first query to see what's available
- For multi-step analysis, make multiple tool calls and synthesize results

CLARIFICATION PROTOCOL:
If the query is unclear:
- Ask what specific data they need
- Ask for timeframes if dates are ambiguous
- Ask which bank if not specified
- Provide examples of available data

DO NOT:
- Pass vague text directly to tools
- Make wild guesses about column values
- Skip necessary filters
- Assume column names

Remember: You are the intelligent layer that translates natural language into structured database queries!


TOOL PARAMETERS (all you need to construct any query):

1. **operation**: "SELECT" (query data) or "DESCRIBE" (get table schema)

2. **tables**: Which table(s) to query
   - Single: "ffiec"
   - Multiple with aliases: [{{"name": "customers", "alias": "c"}}, {{"name": "orders", "alias": "o"}}]

3. **columns**: What to select (optional, defaults to all columns)
   - ["Bank Name", "Value"]
   - ["c.Customer Name", "o.Total"]
   - [{{"column": "Value", "alias": "TotalValue"}}]

4. **filters**: WHERE conditions (optional)
   - [{{"column": "Bank Name", "operator": "=", "value": "Metro Financial"}}]
   - [{{"column": "Value", "operator": ">", "value": 1000000}}]
   - [{{"column": "Status", "operator": "IN", "value": ["Active", "Pending"]}}]
   - Operators: =, !=, >, <, >=, <=, LIKE, IN, BETWEEN, IS NULL, IS NOT NULL

5. **joins**: Join multiple tables (optional)
   - [{{"type": "INNER", "table": {{"name": "orders", "alias": "o"}}, "on": "c.Customer ID = o.Customer ID"}}]

6. **aggregations**: SUM, AVG, COUNT, MIN, MAX (optional)
   - [{{"function": "SUM", "column": "Value", "alias": "Total"}}]
   - [{{"function": "COUNT", "column": "*", "alias": "Count"}}]

7. **group_by**: Group results (optional)
   - ["Bank Name", "Reporting Date"]

8. **order_by**: Sort results (optional)
   - [{{"column": "Value", "direction": "DESC"}}]

9. **having**: Filter grouped results (optional, same format as filters)

10. **limit**: Max rows (optional, default: 100)

HOW TO USE THE TOOL:

For ANY user query, you need to:
1. **Understand** what the user wants
2. **Identify** which tables, columns, and filters are needed
3. **Construct** the appropriate tool parameters
4. **Call** execute_sql_query with those parameters

EXAMPLE SCENARIOS:

Simple Query:
User: "Show me banks with total assets"
You call: execute_sql_query(
    operation="SELECT",
    tables="ffiec",
    columns=["Bank Name", "Value"],
    filters=[{{"column": "Report Element", "operator": "=", "value": "TOTAL ASSETS"}}]
)

Aggregation:
User: "Sum all values by bank name"
You call: execute_sql_query(
    operation="SELECT",
    tables="ffiec",
    group_by=["Bank Name"],
    aggregations=[{{"function": "SUM", "column": "Value", "alias": "Total"}}]
)

Multi-table Join:
User: "Show customers with their orders"
You call: execute_sql_query(
    operation="SELECT",
    tables=[{{"name": "customers", "alias": "c"}}],
    columns=["c.Customer Name", "o.Order Date", "o.Total"],
    joins=[{{"type": "INNER", "table": {{"name": "orders", "alias": "o"}}, "on": "c.Customer ID = o.Customer ID"}}]
)

Complex Query with Multiple Conditions:
User: "Find banks with assets over 10 billion, group by bank, show total and average"
You call: execute_sql_query(
    operation="SELECT",
    tables="ffiec",
    columns=["Bank Name"],
    filters=[{{"column": "Report Element", "operator": "=", "value": "TOTAL ASSETS"}}],
    group_by=["Bank Name"],
    aggregations=[
        {{"function": "SUM", "column": "Value", "alias": "Total Assets"}},
        {{"function": "AVG", "column": "Value", "alias": "Average Assets"}}
    ],
    having=[{{"column": "SUM(Value)", "operator": ">", "value": 10000000000}}]
)

Get Table Schema (when unsure about structure):
You call: execute_sql_query(
    operation="DESCRIBE",
    tables="ffiec"
)

MULTI-STEP QUERIES:

For complex queries requiring multiple steps:
1. First call may get intermediate data
2. Analyze results
3. Make second call based on first results
4. Synthesize final answer

Example:
User: "Compare banks with 'Metro' in name vs others"
Step 1: Get Metro banks data
Step 2: Get non-Metro banks data  
Step 3: Compare and present

IMPORTANT GUIDELINES:

1. **Extract filters from user query**: Look for specific values, comparisons, date ranges, etc.
2. **Determine columns needed**: Only select columns relevant to the query
3. **Identify aggregations**: Look for keywords like "total", "average", "count", "sum"
4. **Handle dates**: Extract date filters from queries like "in 2024" or "on 2024-12-31"
5. **Use LIKE for partial matches**: When user says "banks with Metro" use LIKE operator
6. **Choose appropriate operators**: 
   - Use "=" for exact matches
   - Use ">" "<" for comparisons
   - Use "IN" for multiple values
   - Use "LIKE" for partial text matching
7. **Group intelligently**: If user asks for "by bank" or "per category", use group_by
8. **Sort results**: If user wants "top", "highest", "lowest", use order_by

CLARIFICATION PROTOCOL:

If the query is unclear:
- Ask specific questions about what data is needed
- Use DESCRIBE operation to show available columns
- Provide examples of what you can do

DO NOT:
- Make assumptions about column values
- Skip necessary filters
- Use hardcoded values unless explicitly stated by user

Remember: You have ONE powerful tool that can do EVERYTHING. Focus on translating the user's intent into the correct parameter combination!"""


CLARIFICATION_PROMPT = """You are analyzing whether a user query has sufficient information to query the database.

Available tables: {available_tables}

Evaluate the user's query and determine if it's:
1. COMPLETE - Has all necessary information to execute queries
2. INCOMPLETE - Missing critical information (table identification, filters, etc.)
3. EXPLORATORY - User wants to discover what's available
4. CLARIFICATION - User is providing additional information

Consider:
- Does the query specify which table(s) to use from available tables?
- Does it specify what data/metrics are needed?
- For filtered queries, are necessary identifiers provided?
- Is this a follow-up providing clarification?

Respond with a JSON object (no markdown, just the JSON):
{{
    "status": "COMPLETE" | "INCOMPLETE" | "EXPLORATORY" | "CLARIFICATION",
    "missing_info": ["list", "of", "missing", "items"],
    "suggested_tables": ["tables", "that", "might", "be", "relevant"],
    "suggestions": ["helpful suggestions for user"],
    "can_proceed": true | false,
    "reasoning": "brief explanation"
}}

User query: {query}

Previous context (if any): {context}"""


# ============================================================================
# Agent Graph Builder
# ============================================================================

class SQLAgentGraph:
    """LangGraph workflow for SQL agent with table-aware processing"""
    
    def __init__(
        self, 
        llm_wrapper: LLMWrapper, 
        tools: List,
        table_schemas: Dict[str, Any]
    ):
        """
        Initialize agent graph
        
        Args:
            llm_wrapper: LLM wrapper instance
            tools: List of tool instances
            table_schemas: Dictionary of all table schemas
        """
        self.llm_wrapper = llm_wrapper
        self.tools = tools
        self.table_schemas = table_schemas
        self.graph = None
        
        logger.info(f"Initialized SQLAgentGraph with {len(tools)} generic tools")
    
    def should_continue(self, state: AgentState) -> Literal["tools", "clarify", END]:
        """Determine next step based on current state"""
        messages = state["messages"]
        last_message = messages[-1]
        
        if state.get("awaiting_clarification", False) and isinstance(last_message, AIMessage):
            logger.debug("Awaiting clarification - ending workflow")
            return END
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.debug(f"Tool calls detected: {len(last_message.tool_calls)}")
            return "tools"
        
        if isinstance(last_message, AIMessage):
            content = last_message.content.lower()
            clarification_indicators = [
                "need more information",
                "need clarification",
                "which table",
                "what specific",
                "can you specify",
                "please provide"
            ]
            
            if any(indicator in content for indicator in clarification_indicators):
                logger.debug("Clarification needed")
                return "clarify"
        
        logger.debug("Workflow complete")
        return END
    
    def analyze_with_llm(self, state: AgentState) -> AgentState:
        """Use LLM to analyze if query needs clarification"""
        messages = state["messages"]
        available_tables = state.get("available_tables", [])
        
        user_messages = [m for m in messages if isinstance(m, HumanMessage)]
        if not user_messages:
            return state
        
        last_user_message = user_messages[-1].content
        logger.info(f"Analyzing query for tables {available_tables}: {last_user_message[:100]}...")
        
        context = ""
        if state.get("original_question"):
            context = f"Original question: {state['original_question']}"
            if state.get("clarification_context"):
                context += f"\nContext: {json.dumps(state['clarification_context'])}"
        
        llm = self.llm_wrapper.get_llm()
        analysis_prompt = CLARIFICATION_PROMPT.format(
            available_tables=", ".join(available_tables) if available_tables else "all tables",
            query=last_user_message,
            context=context if context else "None"
        )
        
        try:
            response = llm.invoke([HumanMessage(content=analysis_prompt)])
            logger.debug(f"LLM analysis response received")
            
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
            logger.info(f"Analysis result: {analysis.get('status')}")
            
            if analysis["status"] == "CLARIFICATION" and state.get("original_question"):
                state["awaiting_clarification"] = False
                logger.debug("User providing clarification")
            elif analysis["status"] == "INCOMPLETE":
                if not state.get("original_question"):
                    state["original_question"] = last_user_message
                state["awaiting_clarification"] = True
                state["clarification_context"] = analysis
                logger.info(f"Query incomplete. Missing: {analysis.get('missing_info')}")
            elif analysis["status"] in ["COMPLETE", "EXPLORATORY"]:
                state["awaiting_clarification"] = False
                logger.debug(f"Query is {analysis['status'].lower()}")
            
        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error: {je}")
            state["awaiting_clarification"] = False
        except Exception as e:
            logger.error(f"Error in analysis: {e}", exc_info=True)
            state["awaiting_clarification"] = False
        
        return state
    
    def call_model(self, state: AgentState) -> AgentState:
        """Call the LLM model with context awareness"""
        messages = state["messages"]
        available_tables = state.get("available_tables", [])
        
        # Create system prompt with available tables
        system_prompt = create_system_prompt(self.table_schemas, available_tables)
        
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages
        else:
            # Update system message with current tables
            messages[0] = SystemMessage(content=system_prompt)
        
        if state.get("original_question") and state.get("awaiting_clarification"):
            user_messages = [m for m in messages if isinstance(m, HumanMessage)]
            if user_messages:
                context_msg = SystemMessage(
                    content=f"""CONTEXT: The user originally asked: "{state['original_question']}"
They are now providing clarification. Combine both pieces of information to answer their question."""
                )
                messages = messages[:-1] + [context_msg, messages[-1]]
                logger.debug("Added clarification context")
        
        llm = self.llm_wrapper.get_llm()
        llm_with_tools = llm.bind_tools(self.tools)
        
        logger.debug("Invoking LLM with tools")
        response = llm_with_tools.invoke(messages)
        logger.debug(f"LLM response received")
        
        return {"messages": [response]}
    
    def request_clarification(self, state: AgentState) -> AgentState:
        """Handle clarification request"""
        state["awaiting_clarification"] = True
        logger.info("Clarification requested")
        return state
    
    def build(self) -> StateGraph:
        """Build and compile the agent graph"""
        logger.info("Building agent graph...")
        workflow = StateGraph(AgentState)
        
        workflow.add_node("analyze", self.analyze_with_llm)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("clarify", self.request_clarification)
        
        workflow.set_entry_point("analyze")
        
        workflow.add_edge("analyze", "agent")
        workflow.add_edge("clarify", END)
        
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools",
                "clarify": "clarify",
                END: END
            }
        )
        
        workflow.add_edge("tools", "agent")
        
        self.graph = workflow.compile()
        logger.info("Agent graph compiled")
        return self.graph
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the compiled graph"""
        if self.graph is None:
            self.build()
        
        logger.info(f"Invoking agent with tables: {state.get('available_tables', [])}")
        result = self.graph.invoke(state)
        logger.info("Agent invocation complete")
        return result