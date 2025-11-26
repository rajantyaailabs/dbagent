import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine

from api.router import router
from config import AppConfig, TableConfig, setup_logging
from core import state
from database_tools import create_database_tools

setup_logging(AppConfig.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("SQL Agent API v6.0 - Starting")
    
    # Initialize database
    state.engine = create_engine(AppConfig.DATABASE_URL)
    logger.info("Database engine created")
    
    state.tools = create_database_tools(state.engine)
    logger.info(f"Created {len(state.tools)} table-specific tools")
    
    state.all_table_schemas = TableConfig.get_all_schemas()
    logger.info(f"Loaded {len(state.all_table_schemas)} table schemas")
    
    logger.info(f"Available tables: {list(state.all_table_schemas.keys())}")
    logger.info(f"Default model: {AppConfig.DEFAULT_MODEL}")
    
    yield
    
    logger.info("SQL Agent API - Shutting down")
    if state.engine:
        state.engine.dispose()
    logger.info("Database connections closed")


app = FastAPI(
    title="LangGraph SQL Agent API",
    description="Intelligent multi-table SQL agent with dynamic table selection",
    version="6.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LangGraph SQL Agent API - Dynamic Multi-Table Support",
        "version": "6.0.0",
        "endpoints": {
            "POST /unified": "Invoke agent with optional tables parameter",
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


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {AppConfig.HOST}:{AppConfig.PORT}")

    uvicorn.run(app, host=AppConfig.HOST, port=AppConfig.PORT, log_level=AppConfig.LOG_LEVEL.lower())
