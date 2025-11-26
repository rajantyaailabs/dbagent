import logging
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage

from config import AppConfig
from core import state
from llm_wrapper import LLMWrapper
from workflow import SQLAgentGraph

logger = logging.getLogger(__name__)

def get_or_create_agent(model_provider: str, model_name: str, temperature: float = 0.0) -> SQLAgentGraph:
    """Get or create agent with specified model (with caching)"""
    cache_key = f"{model_name}_{temperature}"

    if cache_key not in state.agent_cache:
        logger.info(f"Creating new agent for model: {model_name}")
        llm_wrapper = LLMWrapper(model_provider=model_provider, model_name=model_name, temperature=temperature)
        agent_graph = SQLAgentGraph(llm_wrapper, state.tools, state.all_table_schemas)
        agent_graph.build()
        state.agent_cache[cache_key] = agent_graph
        logger.debug(f"Agent cached: {cache_key}")
    else:
        logger.debug(f"Using cached agent: {cache_key}")

    return state.agent_cache[cache_key]


def get_or_create_session(session_id: str) -> Dict[str, Any]:
    """Get or create session state"""
    if session_id not in state.conversation_sessions:
        state.conversation_sessions[session_id] = {
            "messages": [],
            "original_question": None,
            "awaiting_clarification": False,
            "clarification_context": None,
            "available_tables": [],
        }
        logger.debug(f"Created session: {session_id}")
    return state.conversation_sessions[session_id]


def normalize_message_content(content: Any) -> str:
    """Normalize message content from different LLM formats to a plain string."""
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


def extract_tables_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[str]:
    """Extract unique table names from tool call arguments"""
    tables = set()
    
    # FFIEC tools that implicitly use the 'ffiec' table
    ffiec_tools = {"query_ffiec_data", "calculate_and_aggregate", "get_ffiec_schema_info"}

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {})

        # Check for FFIEC tools
        if name in ffiec_tools:
            tables.add("ffiec")

        # Check for table_name argument
        if "table_name" in args:
            tables.add(args["table_name"])

        # Check for tables list (for join operations)
        if "tables" in args:
            if isinstance(args["tables"], list):
                for t in args["tables"]:
                    if isinstance(t, dict) and "name" in t:
                        tables.add(t["name"])
                    elif isinstance(t, str):
                        tables.add(t)

    return list(tables)
