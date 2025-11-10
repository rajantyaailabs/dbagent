import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from config import AppConfig, TableConfig, setup_logging
from database_tools import create_database_tools
from llm_wrapper import (
    LLM_API_MAPPING,
    LLMWrapper,
    get_available_models,
    validate_model,
)
from workflow import SQLAgentGraph

# Setup logging
setup_logging(AppConfig.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ============================================================================
# Initialization
# ============================================================================

logger.info("Initializing SQL Agent API...")

# Initialize database
engine = create_engine(AppConfig.DATABASE_URL)
logger.info("Database engine created")

tools = create_database_tools(engine)
logger.info(f"Created {len(tools)} table-specific tools")

# Get all available table schemas
all_table_schemas = TableConfig.get_all_schemas()
logger.info(f"Loaded {len(all_table_schemas)} table schemas")

# Session storage
conversation_sessions: Dict[str, Dict[str, Any]] = {}
logger.info("Session storage initialized")

# Agent cache (model_tables_key -> agent)
agent_cache: Dict[str, SQLAgentGraph] = {}


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="LangGraph SQL Agent API",
    description="Intelligent multi-table SQL agent with dynamic table selection",
    version="6.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("FastAPI application initialized")


# ============================================================================
# Request/Response Models
# ============================================================================


class InvokeRequest(BaseModel):
    message: str
    tables: Optional[List[str]] = None  # NEW: List of tables to query
    session_id: Optional[str] = None
    model: Optional[str] = "Gemini 1.5 Flash"
    temperature: Optional[float] = 0.0
    config: Dict[str, Any] = {}


class InvokeResponse(BaseModel):
    response: str
    tool_calls: List[Dict[str, Any]] = []
    tables_queried: List[str] = []  # Which tables were actually accessed
    awaiting_clarification: bool = False
    original_question: Optional[str] = None
    session_id: Optional[str] = None
    model_used: str = ""
    success: bool = True


class InspectResponse(BaseModel):
    available_models: Dict[str, List[str]]
    model_mapping: Dict[str, Dict[str, str]]
    available_tools: List[Dict[str, str]]
    all_tables: List[str]
    table_schemas: Dict[str, Any]
    database_url: str
    features: List[str]


class ModelListResponse(BaseModel):
    models: Dict[str, List[str]]
    total_count: int


class TableListResponse(BaseModel):
    tables: List[Dict[str, Any]]
    count: int


# ============================================================================
# Helper Functions
# ============================================================================


def get_or_create_agent(model: str, temperature: float = 0.0) -> SQLAgentGraph:
    """Get or create agent with specified model (with caching)"""
    cache_key = f"{model}_{temperature}"

    if cache_key not in agent_cache:
        logger.info(f"Creating new agent for model: {model}")
        llm_wrapper = LLMWrapper(model_name=model, temperature=temperature)
        agent_graph = SQLAgentGraph(llm_wrapper, tools, all_table_schemas)
        agent_graph.build()
        agent_cache[cache_key] = agent_graph
        logger.debug(f"Agent cached: {cache_key}")
    else:
        logger.debug(f"Using cached agent: {cache_key}")

    return agent_cache[cache_key]


def get_or_create_session(session_id: str) -> Dict[str, Any]:
    """Get or create session state"""
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = {
            "messages": [],
            "original_question": None,
            "awaiting_clarification": False,
            "clarification_context": None,
            "available_tables": [],
        }
        logger.debug(f"Created session: {session_id}")
    return conversation_sessions[session_id]


def extract_tables_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[str]:
    """Extract unique table names from tool call arguments"""
    tables = set()

    for tc in tool_calls:
        args = tc.get("args", {})

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


# ============================================================================
# API Endpoints
# ============================================================================


@app.post("/invoke", response_model=InvokeResponse)
async def invoke_agent(request: InvokeRequest):
    """
    Invoke the agent with a user message

    Args:
        request: Contains message, optional tables list, session_id, model, temperature

    Returns:
        Agent response with tool calls and tables accessed
    """
    try:
        logger.info(f"Request: {request.message[:100]}...")
        logger.info(f"Tables: {request.tables}, Model: {request.model}")

        # Validate model
        if request.model and not validate_model(request.model):
            logger.warning(f"Invalid model: {request.model}")
            return InvokeResponse(
                response=f"Invalid model: {request.model}. Use /models endpoint.",
                success=False,
                model_used=request.model or "Unknown",
            )

        # Determine available tables
        if request.tables:
            available_tables = request.tables
            logger.info(f"Using specified tables: {available_tables}")
        else:
            # If no tables specified, use all configured tables
            available_tables = list(all_table_schemas.keys())
            logger.info(f"No tables specified, using all: {available_tables}")

        # Validate tables exist
        invalid_tables = [t for t in available_tables if t not in all_table_schemas]
        if invalid_tables:
            logger.warning(f"Invalid tables: {invalid_tables}")
            return InvokeResponse(
                response=f"Invalid tables: {invalid_tables}. Available tables: {list(all_table_schemas.keys())}",
                success=False,
                model_used=request.model or AppConfig.DEFAULT_MODEL,
            )

        # Session management
        session_id = request.session_id or f"session_{datetime.now().timestamp()}"
        session_state = get_or_create_session(session_id)

        # Update available tables in session
        session_state["available_tables"] = available_tables

        # Add user message
        session_state["messages"].append(HumanMessage(content=request.message))
        logger.debug(f"Messages in session: {len(session_state['messages'])}")

        # Create/get agent
        agent = get_or_create_agent(model=request.model or AppConfig.DEFAULT_MODEL, temperature=request.temperature)

        # Invoke agent with available tables
        logger.info("Invoking agent...")
        result = agent.invoke(
            {
                "messages": session_state["messages"],
                "original_question": session_state.get("original_question"),
                "awaiting_clarification": session_state.get("awaiting_clarification", False),
                "clarification_context": session_state.get("clarification_context"),
                "available_tables": available_tables,
            }
        )
        logger.info("Agent invocation successful")

        # Update session
        session_state["messages"] = result.get("messages", [])
        session_state["original_question"] = result.get("original_question")
        session_state["awaiting_clarification"] = result.get("awaiting_clarification", False)
        session_state["clarification_context"] = result.get("clarification_context")

        # Extract response
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None

        if not last_message:
            raise HTTPException(status_code=500, detail="No response from agent")

        # Extract tool calls
        tool_calls = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({"name": tc.get("name", ""), "args": tc.get("args", {})})

        # Determine which tables were actually queried
        tables_queried = extract_tables_from_tool_calls(tool_calls)

        logger.info(f"Tool calls: {len(tool_calls)}, Tables queried: {tables_queried}")

        response_text = last_message.content if hasattr(last_message, "content") else str(last_message)

        return InvokeResponse(
            response=response_text,
            tool_calls=tool_calls,
            tables_queried=tables_queried,
            awaiting_clarification=result.get("awaiting_clarification", False),
            original_question=result.get("original_question"),
            session_id=session_id,
            model_used=request.model or AppConfig.DEFAULT_MODEL,
            success=True,
        )

    except Exception as e:
        logger.error(f"Error in invoke_agent: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}\n\n{traceback.format_exc()}")


@app.get("/models", response_model=ModelListResponse)
async def list_models():
    """Get list of available LLM models"""
    logger.debug("Listing models")
    models = get_available_models()
    return ModelListResponse(models=models, total_count=sum(len(v) for v in models.values()))


@app.get("/tables", response_model=TableListResponse)
async def list_tables():
    """Get list of all available tables"""
    logger.debug("Listing tables")

    tables_info = []
    for table_name, schema in all_table_schemas.items():
        tables_info.append(
            {
                "name": table_name,
                "description": schema.description,
                "columns": schema.columns,
                "column_count": len(schema.columns),
            }
        )

    return TableListResponse(tables=tables_info, count=len(tables_info))


@app.get("/inspect", response_model=InspectResponse)
async def inspect_agent():
    """Get comprehensive agent configuration"""
    logger.debug("Inspecting configuration")

    tools_info = [{"name": tool.name, "description": tool.description, "args": str(tool.args)} for tool in tools]

    schemas_response = {}
    for table_name, schema in all_table_schemas.items():
        schemas_response[table_name] = {"description": schema.description, "columns": schema.columns}

    features = [
        "Dynamic table selection per request",
        "Generic tools work with any table",
        "Multi-table query support with automatic joining",
        "Intelligent tool selection by LLM",
        "Multi-LLM support (OpenAI, Google, Anthropic)",
        "Context-aware clarification",
        "Session-based memory",
        "Comprehensive logging",
    ]

    masked_url = AppConfig.DATABASE_URL
    if "@" in masked_url:
        masked_url = masked_url.replace(masked_url.split("@")[0].split("//")[1], "***:***")

    return InspectResponse(
        available_models=get_available_models(),
        model_mapping=LLM_API_MAPPING,
        available_tools=tools_info,
        all_tables=list(all_table_schemas.keys()),
        table_schemas=schemas_response,
        database_url=masked_url,
        features=features,
    )


@app.post("/reset_session")
async def reset_session(session_id: str):
    """Reset a conversation session"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
        logger.info(f"Session reset: {session_id}")
        return {"message": f"Session {session_id} reset", "success": True}

    logger.warning(f"Session not found: {session_id}")
    return {"message": f"Session {session_id} not found", "success": False}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.debug("Health check passed")
        return {
            "status": "healthy",
            "database": "connected",
            "available_tables": list(all_table_schemas.keys()),
            "tools_count": len(tools),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LangGraph SQL Agent API - Dynamic Multi-Table Support",
        "version": "6.0.0",
        "endpoints": {
            "POST /invoke": "Invoke agent with optional tables parameter",
            "GET /models": "List available LLM models",
            "GET /tables": "List all database tables",
            "GET /inspect": "Get detailed configuration",
            "POST /reset_session": "Reset conversation session",
            "GET /health": "Health check",
        },
        "key_features": [
            "Specify tables per request via 'tables' parameter",
            "5 generic tools work with any table",
            "Automatic multi-table query handling",
            "Smart tool selection by LLM",
        ],
        "example_requests": {
            "single_table": {"message": "Show me all data from Metro Financial", "tables": ["ffiec"]},
            "multi_table": {"message": "Show customers with their orders", "tables": ["customers", "orders"]},
            "all_tables": {"message": "Search for 'Metro' across all tables", "tables": None},
        },
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("=" * 60)
    logger.info("SQL Agent API v6.0 - Starting")
    logger.info("=" * 60)
    logger.info(f"Available tables: {list(all_table_schemas.keys())}")
    logger.info(f"Generic tools: {len(tools)}")
    logger.info(f"Default model: {AppConfig.DEFAULT_MODEL}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("SQL Agent API - Shutting down")
    engine.dispose()
    logger.info("Database connections closed")


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {AppConfig.HOST}:{AppConfig.PORT}")

    uvicorn.run(app, host=AppConfig.HOST, port=AppConfig.PORT, log_level=AppConfig.LOG_LEVEL.lower())
