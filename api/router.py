import logging
import traceback
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from api.schemas import (
    InvokeRequest,
    InvokeResponse,
    InspectResponse,
    ModelListResponse,
    TableListResponse,
)
from config import AppConfig
from core import state
from llm_wrapper import LLM_API_MAPPING, get_available_models, validate_model
from services.agent_service import (
    get_or_create_agent,
    get_or_create_session,
    normalize_message_content,
    extract_tables_from_tool_calls,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/unified", response_model=InvokeResponse)
async def invoke_agent(request: InvokeRequest):
    """Invoke the agent with a user message"""
    try:
        logger.info(f"Request: {request.message[:100]}...")
        logger.info(f"Tables: {request.tables}, Model: {request.model_name}")

        # Validate model
        if request.model_name and not validate_model(request.model_name):
            logger.warning(f"Invalid model: {request.model_name}")
            return InvokeResponse(
                response=f"Invalid model: {request.model_name}. Use /models endpoint.",
                success=False,
                model_used=request.model_name or "Unknown",
            )

        # Determine available tables
        if request.tables:
            available_tables = request.tables
            logger.info(f"Using specified tables: {available_tables}")
        else:
            # If no tables specified, use all configured tables
            available_tables = list(state.all_table_schemas.keys())
            logger.info(f"No tables specified, using all: {available_tables}")

        # Validate tables exist
        invalid_tables = [t for t in available_tables if t not in state.all_table_schemas]
        if invalid_tables:
            logger.warning(f"Invalid tables: {invalid_tables}")
            return InvokeResponse(
                response=f"Invalid tables: {invalid_tables}. Available tables: {list(state.all_table_schemas.keys())}",
                success=False,
                model_used=request.model_name or AppConfig.DEFAULT_MODEL,
            )

        # Session management
        session_id = request.session_id or f"session_{datetime.now().timestamp()}"
        session_state = get_or_create_session(session_id)

        # Update available tables in session
        session_state["available_tables"] = available_tables

        # Add user message
        session_state["messages"].append(HumanMessage(content=request.message))
        
        # Create/get agent
        agent = get_or_create_agent(model_provider=request.model_provider,
                                    model_name=request.model_name or AppConfig.DEFAULT_MODEL,
                                    temperature=request.temperature)

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

        # Normalize response content to handle different LLM formats
        raw_content = last_message.content if hasattr(last_message, "content") else str(last_message)
        response_text = normalize_message_content(raw_content)

        return InvokeResponse(
            response=response_text,
            tool_calls=tool_calls,
            tables_queried=tables_queried,
            awaiting_clarification=result.get("awaiting_clarification", False),
            original_question=result.get("original_question"),
            session_id=session_id,
            model_used=request.model_name or AppConfig.DEFAULT_MODEL,
            success=True,
        )

    except Exception as e:
        logger.error(f"Error in invoke_agent: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}\n\n{traceback.format_exc()}")


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """Get list of available LLM models"""
    logger.debug("Listing models")
    models = get_available_models()
    return ModelListResponse(models=models, total_count=sum(len(v) for v in models.values()))


@router.get("/tables", response_model=TableListResponse)
async def list_tables():
    """Get list of all available tables"""
    logger.debug("Listing tables")

    tables_info = []
    for table_name, schema in state.all_table_schemas.items():
        tables_info.append(
            {
                "name": table_name,
                "description": schema.description,
                "columns": schema.columns,
                "column_count": len(schema.columns),
            }
        )

    return TableListResponse(tables=tables_info, count=len(tables_info))


@router.get("/inspect", response_model=InspectResponse)
async def inspect_agent():
    """Get comprehensive agent configuration"""
    logger.debug("Inspecting configuration")

    tools_info = [{"name": tool.name, "description": tool.description, "args": str(tool.args)} for tool in state.tools]

    schemas_response = {}
    for table_name, schema in state.all_table_schemas.items():
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
        all_tables=list(state.all_table_schemas.keys()),
        table_schemas=schemas_response,
        database_url=masked_url,
        features=features,
    )


@router.post("/reset_session")
async def reset_session(session_id: str):
    """Reset a conversation session"""
    if session_id in state.conversation_sessions:
        del state.conversation_sessions[session_id]
        logger.info(f"Session reset: {session_id}")
        return {"message": f"Session {session_id} reset", "success": True}

    logger.warning(f"Session not found: {session_id}")
    return {"message": f"Session {session_id} not found", "success": False}


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        with state.engine.connect() as conn:
            conn.execute("SELECT 1")

        logger.debug("Health check passed")
        return {
            "status": "healthy",
            "database": "connected",
            "available_tables": list(state.all_table_schemas.keys()),
            "tools_count": len(state.tools),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}
