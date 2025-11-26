from typing import Any, Dict, List
from sqlalchemy.engine import Engine
from workflow import SQLAgentGraph

# Global state variables
engine: Engine = None
tools: List = []
all_table_schemas: Dict[str, Any] = {}
conversation_sessions: Dict[str, Dict[str, Any]] = {}
agent_cache: Dict[str, SQLAgentGraph] = {}
