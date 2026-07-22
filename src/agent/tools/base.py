# ruff: noqa: E501
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """
    The abstract base class for all Agent tools.
    Enforces a strict contract so the Orchestrator can dynamically load
    and execute any tool (semantic search, AST reading, etc.) blindly.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The exact string the LLM must output in <name> to call this tool."""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """The instruction prompt injected into the LLM context explaining when to use this tool."""
        pass

    @abstractmethod
    def execute(self, args: Dict[str, Any]) -> str:
        """
        Executes the tool's core logic. 
        Returns a raw string (e.g., code snippets, error traces) which is saved to the State Clipboard.
        """
        pass
        
    @abstractmethod
    def get_function_declaration(self) -> Dict[str, Any]:
        """
        Returns the JSON schema for native LLM function calling APIs (Gemini/OpenAI).
        Each tool must define its own schema matching its specific parameters.
        """
        pass
