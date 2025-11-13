"""
Workflow - LangGraph workflow for SQL agent with table-aware clarification
"""

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
# Utility Functions
# ============================================================================

def normalize_content(content: Any) -> str:
    """
    Normalize message content from different LLM formats to a plain string.

    Different LLMs return content in different formats:
    - OpenAI: Returns plain string
    - Gemini: Returns list of content blocks [{'type': 'text', 'text': '...'}]
    - Anthropic: Can return either format

    Args:
        content: Content from LLM message (string, list, or other)

    Returns:
        Normalized string content
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    text_parts.append(str(block["text"]))
                elif "content" in block:
                    text_parts.append(str(block["content"]))
                else:
                    text_parts.append(str(block))
            elif isinstance(block, str):
                text_parts.append(block)
            else:
                text_parts.append(str(block))
        return " ".join(text_parts).strip()

    return str(content)


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
    
    return f"""You are a highly intelligent database assistant specialized in querying banking and financial regulatory data.

AVAILABLE TABLES FOR THIS REQUEST:
{tables_section}

AVAILABLE TOOLS FOR FFIEC TABLE:
You have 3 powerful, flexible tools:

1. **query_ffiec_data** - Primary query tool with flexible filtering
   Handles: Total assets, capital structure, balance sheet items, RWA, schedule comparisons, pattern matching
   Use flexible filters: bank_identifier, reporting_date, report_elements, schedules, source_item_pattern

2. **calculate_and_aggregate** - Calculations and aggregations
   Handles: Sums, averages, counts, capital ratios, grouping
   Supports: tier1_ratio calculation, custom aggregations, multi-level grouping

3. **get_ffiec_schema_info** - Schema discovery
   Handles: Available banks, dates, report elements, schedules, table overview

TOOL SELECTION STRATEGY:

Query Type → Tool Selection:

- "Total assets for Bank X" → query_ffiec_data(bank_identifier="X", report_elements=["TOTAL ASSETS"])
- "Capital structure" → query_ffiec_data(bank_identifier="X", source_item_pattern="%stock%|%surplus%|%earnings%")
- "Verify Tier 1 ratio" → calculate_and_aggregate(calculation_type="tier1_ratio")
- "Sum by category" → calculate_and_aggregate(calculation_type="sum", group_by=["Report Element"])
- "Schedule RC items" → query_ffiec_data(schedules=["RC"])
- "Compare RC and RC-R" → query_ffiec_data(schedules=["RC", "RC-R"])
- "What banks exist?" → get_ffiec_schema_info(info_type="banks")

KEY FILTERING PARAMETERS:

**query_ffiec_data filters:**
- bank_identifier: "Metro Financial Corporation" or "BANK001"
- identifier_type: "name" or "id"
- reporting_date: "2024-12-31"
- report_elements: ["TOTAL ASSETS"] or ["TOTAL ASSETS", "CAPITAL"]
- schedules: ["RC"] or ["RC", "RC-R"]
- source_item_pattern: "%Cash%" or "%stock%|%surplus%" (use | for OR)
- columns: ["Bank Name", "Value"] (None for all)

**calculate_and_aggregate types:**
- "sum": Sum values
- "avg": Average values
- "count": Count records
- "tier1_ratio": Calculate Tier 1 capital ratio (auto-verifies formula)

EXAMPLES:

1. Total assets: query_ffiec_data(bank_identifier="Metro Financial Corporation", reporting_date="2024-12-31", report_elements=["TOTAL ASSETS"])

2. Capital structure: query_ffiec_data(bank_identifier="BANK001", identifier_type="id", reporting_date="2024-12-31", source_item_pattern="%Common stock%|%Surplus%|%Retained earnings%")

3. Verify ratio: calculate_and_aggregate(bank_identifier="BANK001", identifier_type="id", reporting_date="2024-12-31", calculation_type="tier1_ratio")

4. Compare schedules: query_ffiec_data(bank_identifier="BANK004", identifier_type="id", reporting_date="2024-12-31", schedules=["RC", "RC-R"])

5. Loan items: query_ffiec_data(bank_identifier="Metro Financial Corporation", source_item_pattern="%loan%")

Remember: These tools are powerful and flexible - use appropriate filters to narrow results!

YOUR RESPONSIBILITY:
Analyze the user's query and select the MOST APPROPRIATE tool(s). Extract structured parameters from natural language.

KEY CONCEPTS:
- **Schedule RC**: Balance Sheet items (assets, liabilities, equity)
- **Schedule RC-R**: Risk-Based Capital (risk-weighted assets, capital ratios)
- **Schedule RC-G**: Off-balance sheet items
- **Report Element**: High-level category (TOTAL ASSETS, CAPITAL, etc.)
- **Source Item**: Detailed line item within a report element
- **FFIEC 101**: Required for banks with $10B+ in total assets

PARAMETER EXTRACTION GUIDELINES:

1. **Bank Identification:**
   - "Metro Financial" → "Metro Financial Corporation"
   - "BANK001" → "BANK001"
   - Always use exact Bank Name or Bank ID

2. **Date Handling:**
   - "December 31, 2024" → "2024-12-31"
   - "Q4 2024" / "year-end 2024" → "2024-12-31"
   - "as of 31-12-2024" → "2024-12-31"

3. **Value Extraction:**
   - User says "over 10 billion" → {{"operator": ">", "value": 10000000000}}
   - User says "at least 1 million" → {{"operator": ">=", "value": 1000000}}

EXAMPLE TOOL SELECTION:

Query: "What is the total asset value for Metro Financial as of Dec 31, 2024?"
Tool: get_bank_total_assets
Parameters: {{"bank_name": "Metro Financial Corporation", "reporting_date": "2024-12-31"}}

Query: "Summarize the capital structure for BANK001"
Tool: get_capital_structure
Parameters: {{"bank_id": "BANK001", "reporting_date": "2024-12-31"}}

Query: "Verify Tier 1 capital ratio for BANK001"
Tool: calculate_capital_ratio
Parameters: {{"bank_id": "BANK001", "reporting_date": "2024-12-31", "ratio_type": "Tier 1"}}

Query: "How does Schedule RC relate to Schedule RC-R for BANK004?"
Tool: compare_schedules
Parameters: {{"bank_id": "BANK004", "schedules": ["RC", "RC-R"], "reporting_date": "2024-12-31"}}

Query: "Show all loan-related items for Metro Financial"
Tool: query_by_report_element
Parameters: {{"bank_name": "Metro Financial Corporation", "report_element_pattern": "%LOAN%"}}

MULTI-TOOL QUERIES:
For complex queries, you may need multiple tool calls:
1. Call first tool to get base data
2. Call second tool for related data
3. Synthesize and compare results

IMPORTANT:
- Extract exact values from user queries
- Use appropriate tool for the scenario
- For capital ratio verification, use calculate_capital_ratio
- For schedule relationships, use compare_schedules
- Always specify reporting date when mentioned
- Convert natural language dates to YYYY-MM-DD format

Remember: Choose the most specific tool that matches the user's intent!"""


CLARIFICATION_PROMPT = """You are analyzing whether a user query has sufficient information to query the database.

Available tables: {available_tables}

Evaluate the user's query and determine if it's:
1. COMPLETE - Has all necessary information
2. INCOMPLETE - Missing critical information
3. EXPLORATORY - User wants to discover what's available
4. CLARIFICATION - User is providing additional information

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
        
        logger.info(f"Initialized SQLAgentGraph with {len(tools)} tools")
    
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
            # Normalize content to handle different LLM formats
            content = normalize_content(last_message.content).lower()

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
    
    def format_tool_response(self, state: AgentState) -> AgentState:
        """Format tool responses into human-readable text"""
        messages = state["messages"]
        
        # Find the last tool message and AI response
        tool_messages = []
        user_question = None
        
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "tool":
                tool_messages.insert(0, msg)
            elif isinstance(msg, HumanMessage):
                user_question = msg.content
                break
        
        if not tool_messages:
            logger.debug("No tool messages to format")
            return state
        
        # Get the last AI message
        last_ai_msg = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_msg = msg
                break
        
        # If the AI message already has string content, skip formatting
        if last_ai_msg and isinstance(last_ai_msg.content, str) and last_ai_msg.content.strip():
            logger.debug("AI response already formatted as string")
            return state
        
        # Extract tool results
        tool_results = []
        for tool_msg in tool_messages:
            tool_results.append({
                "tool_name": tool_msg.name if hasattr(tool_msg, "name") else "unknown",
                "result": tool_msg.content
            })
        
        # Create formatting prompt
        formatting_prompt = f"""You are a helpful assistant that formats database query results into clear, readable responses.

User's Question: {user_question}

Tool Results:
{json.dumps(tool_results, indent=2, default=str)}

Your task:
1. Analyze the tool results
2. Create a clear, concise, human-readable response that answers the user's question
3. Format numbers appropriately (e.g., "$2.52 billion" instead of "2520000000")
4. Organize information logically
5. If multiple items, present them in a clear structure (bullet points, tables, etc.)
6. Highlight key findings

Provide a natural language response that directly answers the user's question based on the tool results."""

        try:
            llm = self.llm_wrapper.get_llm()
            formatted_response = llm.invoke([HumanMessage(content=formatting_prompt)])

            # Normalize formatted response content
            formatted_text = normalize_content(formatted_response.content)

            logger.info(f"Formatted response: {len(formatted_text)} chars")

            # Replace the last AI message with formatted response
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], AIMessage):
                    messages[i] = AIMessage(content=formatted_text)
                    break

            return {"messages": messages}

        except Exception as e:
            logger.error(f"Error formatting response: {e}", exc_info=True)
            return state
    
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