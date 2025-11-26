from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class InvokeRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    message: str
    tables: Optional[List[str]] = None
    session_id: Optional[str] = None
    model_provider: Optional[str] = "google"
    model_name: Optional[str] = "gemini-2.5-flash"
    temperature: Optional[float] = 0.0
    config: Dict[str, Any] = {}


class InvokeResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    response: str
    tool_calls: List[Dict[str, Any]] = []
    tables_queried: List[str] = []
    awaiting_clarification: bool = False
    original_question: Optional[str] = None
    session_id: Optional[str] = None
    model_used: str = ""
    success: bool = True


class InspectResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
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
