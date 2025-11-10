"""
Configuration module for SQL Agent
"""

import logging
import os
from typing import Dict, List

from pydantic import BaseModel


# Logging Configuration
def setup_logging(log_level: str = "INFO"):
    """
    Setup logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from some verbose libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# Table Configuration
class TableSchema(BaseModel):
    """Schema definition for a database table"""

    name: str
    columns: Dict[str, str]  # column_name: column_type
    description: str


class TableConfig:
    """Configuration for database tables"""

    # FFIEC table schema
    FFIEC_SCHEMA = TableSchema(
        name="ffiec",
        columns={
            "Bank ID": "varchar(100)",
            "Bank Name": "varchar(100)",
            "Report Element": "varchar(100)",
            "Item Number": "varchar(100)",
            "Schedule": "varchar(100)",
            "Source Item": "varchar(500)",
            "Source Item Number": "varchar(500)",
            "Value": "bigint",
            "Reporting Date": "date",
        },
        description="FFIEC banking data with financial reports and metrics",
    )

    # Add more table schemas here as needed
    # Example:
    # ANOTHER_TABLE_SCHEMA = TableSchema(
    #     name="another_table",
    #     columns={...},
    #     description="Description of another table"
    # )

    @classmethod
    def get_available_tables(cls) -> List[str]:
        """Get list of available table names"""
        return [
            attr_value.name for attr_name, attr_value in cls.__dict__.items() if isinstance(attr_value, TableSchema)
        ]

    @classmethod
    def get_table_schema(cls, table_name: str) -> TableSchema:
        """Get schema for a specific table"""
        for attr_name, attr_value in cls.__dict__.items():
            if isinstance(attr_value, TableSchema) and attr_value.name == table_name:
                return attr_value
        raise ValueError(f"Table schema not found: {table_name}")

    @classmethod
    def get_all_schemas(cls) -> Dict[str, TableSchema]:
        """Get all table schemas as a dictionary"""
        return {
            attr_value.name: attr_value
            for attr_name, attr_value in cls.__dict__.items()
            if isinstance(attr_value, TableSchema)
        }


# Application Configuration
class AppConfig:
    """Main application configuration"""

    # Environment
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/bankdb")

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Default LLM Settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "Gemini 2.5 Flash")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))

    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Table Configuration
    ENABLED_TABLES: List[str] = os.getenv("ENABLED_TABLES", "ffiec").split(",")

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        errors = []

        if not cls.DATABASE_URL or cls.DATABASE_URL == "postgresql://user:password@localhost:5432/dbname":
            errors.append("DATABASE_URL not configured")

        if not any([cls.OPENAI_API_KEY, cls.GOOGLE_API_KEY, cls.ANTHROPIC_API_KEY]):
            errors.append("At least one LLM API key must be configured")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

        return True
