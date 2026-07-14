# ruff: noqa: E501
import re
from typing import Optional, Tuple


class XMLToolParser:
    """
    Custom XML Parsing scheme for Agent tool selection.
    This guarantees high debuggability and enforces a strict Chain-of-Thought
    <reasoning> step BEFORE the LLM is allowed to write a <tool_call>.
    """
    
    TOOL_PATTERN = re.compile(
        r"<reasoning>(.*?)</reasoning>\s*<tool_call>\s*<name>(.*?)</name>\s*<args>(.*?)</args>\s*</tool_call>", 
        re.DOTALL | re.IGNORECASE
    )
    
    FINAL_ANSWER_PATTERN = re.compile(
        r"<reasoning>(.*?)</reasoning>\s*<final_answer>(.*?)</final_answer>",
        re.DOTALL | re.IGNORECASE
    )

    @classmethod
    def parse_llm_output(cls, raw_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Parses the raw LLM output.
        Returns: (reasoning, tool_name, tool_args_json, final_answer)
        """
        # 1. Did the Agent decide it was DONE?
        final_match = cls.FINAL_ANSWER_PATTERN.search(raw_text)
        if final_match:
            reasoning = final_match.group(1).strip()
            final_answer = final_match.group(2).strip()
            return reasoning, None, None, final_answer
            
        # 2. Did the Agent decide to use a Tool?
        tool_match = cls.TOOL_PATTERN.search(raw_text)
        if tool_match:
            reasoning = tool_match.group(1).strip()
            tool_name = tool_match.group(2).strip()
            tool_args = tool_match.group(3).strip()
            return reasoning, tool_name, tool_args, None
            
        # 3. Formatting Error (LLM hallucinated syntax or forgot XML)
        return None, None, None, None

    @classmethod
    def get_system_prompt_instructions(cls) -> str:
        """The strict formatting rules injected into the Agent's system prompt."""
        return """You must respond using ONE of the following two XML formats.
You MUST write the <reasoning> block first to explain your logic.

OPTION 1: Use a Tool
If you need more information from the codebase, use a tool:
<reasoning>
I need to find how RandomForest validates inputs. I will search the codebase.
</reasoning>
<tool_call>
  <name>search_codebase</name>
  <args>{"query": "RandomForest validate inputs"}</args>
</tool_call>

OPTION 2: Final Answer
If you have gathered enough verified codebase context to answer the user's question fully:
<reasoning>
I have successfully traced the inheritance from RandomForest to check_X_y. I am ready to answer.
</reasoning>
<final_answer>
The input validation is handled by...
</final_answer>
"""
