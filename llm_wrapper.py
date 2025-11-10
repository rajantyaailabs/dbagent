import os
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# LLM Model Configuration
LLM_API_MAPPING = {
    # Fast & Cost-efficient
    "GPT-5 Mini": {"provider": "openai", "model": "gpt-5-mini"},
    "Gemini 2.5 Flash": {"provider": "google", "model": "gemini-2.5-flash"},
    "Gemini 2.0 Flash": {"provider": "google", "model": "gemini-2.0-flash-exp"},
    "Gemini 1.5 Flash": {"provider": "google", "model": "gemini-1.5-flash"},
    
    # Versatile & Highly Intelligent
    "GPT-4.1": {"provider": "openai", "model": "gpt-4.1"},
    "GPT-5 Preview": {"provider": "openai", "model": "gpt-5-preview"},
    "GPT-4o": {"provider": "openai", "model": "gpt-4o"},
    "Claude Sonnet 3.5": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
    "Claude Sonnet 3.7": {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
    "Claude Sonnet 4": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    
    # Most Powerful (Complex Tasks)
    "Claude Sonnet 3.7 Thinking": {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
    "Gemini 2.5 Pro": {"provider": "google", "model": "gemini-2.5-pro"},
    "Gemini 2.0 Flash Thinking": {"provider": "google", "model": "gemini-2.0-flash-thinking-exp"},
}


class LLMWrapper:
    """Unified wrapper for multiple LLM providers"""
    
    def __init__(
        self,
        model_name: str = "Gemini 1.5 Flash",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize LLM wrapper with specified model
        
        Args:
            model_name: Name from LLM_API_MAPPING or direct model string
            temperature: Temperature for generation (0.0 - 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        # Get provider and model from mapping
        if model_name in LLM_API_MAPPING:
            config = LLM_API_MAPPING[model_name]
            self.provider = config["provider"]
            self.model = config["model"]
        else:
            # Assume direct model specification
            self.provider = self._detect_provider(model_name)
            self.model = model_name
        
        self._llm = None
    
    def _detect_provider(self, model: str) -> str:
        """Detect provider from model name"""
        model_lower = model.lower()
        if "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        elif "gemini" in model_lower or "google" in model_lower:
            return "google"
        elif "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        else:
            # Default to OpenAI
            return "openai"
    
    def get_llm(self) -> BaseChatModel:
        """Get or create LLM instance"""
        if self._llm is not None:
            return self._llm
        
        if self.provider == "openai":
            self._llm = self._create_openai_llm()
        elif self.provider == "google":
            self._llm = self._create_google_llm()
        elif self.provider == "anthropic":
            self._llm = self._create_anthropic_llm()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        return self._llm
    
    def _create_openai_llm(self) -> ChatOpenAI:
        """Create OpenAI LLM instance"""
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
        
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        
        params.update(self.kwargs)
        
        return ChatOpenAI(**params)
    
    def _create_google_llm(self) -> ChatGoogleGenerativeAI:
        """Create Google Gemini LLM instance"""
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
        }
        
        if self.max_tokens:
            params["max_output_tokens"] = self.max_tokens
        
        params.update(self.kwargs)
        
        return ChatGoogleGenerativeAI(**params)
    
    def _create_anthropic_llm(self) -> ChatAnthropic:
        """Create Anthropic Claude LLM instance"""
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        }
        
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        
        params.update(self.kwargs)
        
        return ChatAnthropic(**params)
    
    def __str__(self) -> str:
        return f"LLMWrapper(provider={self.provider}, model={self.model})"
    
    def __repr__(self) -> str:
        return self.__str__()


def get_available_models() -> Dict[str, List[str]]:
    """Get available models grouped by category"""
    return {
        "fast": ["GPT-5 Mini", "Gemini 2.0 Flash", "Gemini 1.5 Flash"],
        "versatile": [
            "GPT-4.1", "GPT-5 Preview", "GPT-4o",
            "Claude Sonnet 3.5", "Claude Sonnet 3.7", "Claude Sonnet 4"
        ],
        "powerful": [
            "Claude Sonnet 3.7 Thinking", "Gemini 2.5 Pro", 
            "Gemini 2.0 Flash Thinking"
        ]
    }


def validate_model(model_name: str) -> bool:
    """Validate if model name is supported"""
    return model_name in LLM_API_MAPPING