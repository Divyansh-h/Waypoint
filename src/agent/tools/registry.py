# ruff: noqa: E501
from typing import Any, Dict, List

from agent.tools.base import BaseTool


class ToolRegistry:
    """
    Centralized registry for all Agent tools.
    Decouples tool management from the core Orchestrator loop, allowing easy
    plug-and-play of new tools (e.g. sandbox, git, AST search) without editing
    state machine logic.
    """
    
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        
    def register(self, tool: BaseTool) -> None:
        """Registers a tool by its exact LLM callable name."""
        self._tools[tool.name] = tool
        
    def get_tool(self, name: str) -> BaseTool:
        """Fetches a tool by name, raising KeyError if not found."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]
        
    def get_all_tools(self) -> List[BaseTool]:
        """Returns all registered tools."""
        return list(self._tools.values())
        
    def get_function_declarations(self) -> List[Dict[str, Any]]:
        """Returns the JSON schema definitions for all registered tools."""
        return [tool.get_function_declaration() for tool in self._tools.values()]
        
    def has_tool(self, name: str) -> bool:
        """Checks if a tool exists."""
        return name in self._tools
