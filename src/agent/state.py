# ruff: noqa: E501
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class AgentState(Enum):
    """
    Strict enumeration of valid states for the Agent Orchestrator.
    Prevents silent string-typo routing bugs in the main loop.
    """
    PLANNING = auto()
    RETRIEVING = auto()
    EXECUTING_TOOL = auto()
    EVALUATING = auto()
    SYNTHESIZING = auto()
    DONE = auto()
    FAILED = auto()

@dataclass
class TaskContext:
    """
    The explicitly managed Hybrid-Memory object holding the running context.
    Separates compressed reasoning from lossless raw codebase context.
    """
    # 1. The Core Task
    query: str
    
    # 2. The Planning Scratchpad (Compressed Running Summary)
    # E.g., "Attempted to search 'validate_inputs', failed. Now looking at 'BaseForest'."
    reasoning_history: List[str] = field(default_factory=list)
    
    # 3. The Code Clipboard (Lossless Context)
    # Map of chunk_id -> raw Python string
    gathered_context: Dict[str, str] = field(default_factory=dict)
    
    # 4. Routing metadata
    iterations: int = 0
    retrieval_count: int = 0
    current_state: AgentState = AgentState.PLANNING
    final_answer: Optional[str] = None
    
    def summarize_context_for_prompt(self) -> str:
        """Formats the lossless code chunks into a string for the LLM."""
        if not self.gathered_context:
            return "No codebase context gathered yet."
            
        res = "--- GATHERED CODEBASE CONTEXT ---\n"
        for cid, content in self.gathered_context.items():
            res += f"\n[Snippet ID: {cid}]\n{content}\n"
        return res

@dataclass
class AgentResult:
    """The final payload returned by the Orchestrator."""
    answer: str
    iterations: int
    success: bool
    termination_reason: str
    context_used: Dict[str, str]
