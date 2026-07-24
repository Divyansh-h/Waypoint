# ruff: noqa: E501
import concurrent.futures
import json
import os
import time
from typing import Any, Optional

import yaml

from agent.state import AgentResult, AgentState, TaskContext
from agent.tools.ast_search import ASTSearchTool
from agent.tools.git_patch import GitPatchTool
from agent.tools.registry import ToolRegistry
from agent.tools.sandbox import CodeSandboxTool
from agent.tools.search_codebase import SearchCodebaseTool
from agent.trace import TelemetryTracer
from utils.logger import setup_logger

logger = setup_logger("Orchestrator")

def _execute_with_timeout(func: Any, timeout_seconds: float, *args: Any, **kwargs: Any) -> Any:
    """Executes a function in a background thread, strictly terminating if it exceeds the wall-clock timeout."""
    if timeout_seconds <= 0:
        raise TimeoutError("Wall-clock budget already exhausted.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Execution stalled and was killed after {timeout_seconds:.1f}s")

class AgentOrchestrator:
    """
    The primary state machine loop. Manages transitions between PLANNING,
    RETRIEVING, EVALUATING, and SYNTHESIZING up to a hard iteration limit or timeout.
    Uses Native LLM Function Calling instead of custom parsing.
    """
    def __init__(self, config_path: str = "configs/agent.yaml", llm_client: Any = None, tool_registry: Any = None) -> None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["agent"]
            self.max_iterations = config.get("max_steps", 6)
            self.timeout_seconds = config.get("timeout_seconds", 120)
            self.tool_timeout_seconds = config.get("tool_timeout_seconds", 15)
            self.max_reformulations = config.get("max_reformulations", 3)
            
        self.tracer = TelemetryTracer()
        self.llm_client = llm_client
        
        if tool_registry is None:
            self.tool_registry = ToolRegistry()
            self.tool_registry.register(SearchCodebaseTool())
            self.tool_registry.register(CodeSandboxTool())
            self.tool_registry.register(ASTSearchTool())
            self.tool_registry.register(GitPatchTool())
        else:
            self.tool_registry = tool_registry
        
        self.system_prompt_base = """You are an advanced Agentic Coding Assistant analyzing a complex Python codebase.
Your goal is to answer the user's question accurately by exploring the codebase using the tools provided.

Tool Selection Guidelines:
- `search_codebase`: Use for general semantic queries or conceptual searches.
- `ast_search`: Use for strict structural lookups (definitions, callers, subclasses) when you know the exact name.
- `run_code_sandbox`: Use to securely run Python code to empirically verify array shapes, math, or scikit-learn behaviors on toy data.
- `git_patch`: Use to apply diffs locally, run pytest validation, and draft PR templates for your proposed fixes.

When you have gathered enough verified context and validated your answer, just output the final answer as text.
"""

    def _generate(self, prompt: str, timeout_seconds: float = 120.0) -> Any:
        def _call_llm() -> Any:
            if self.llm_client is not None:
                return self.llm_client.generate_content(prompt)
                
            import google.generativeai as genai
            from google.api_core import exceptions
            if "GEMINI_API_KEY" not in os.environ:
                class MockResp:
                    text = "API KEY MISSING."
                    @property
                    def parts(self) -> Any: return []
                return MockResp()
                
            tool_declaration = {"function_declarations": self.tool_registry.get_function_declarations()}
            model = genai.GenerativeModel(model_name='gemini-2.5-flash', tools=[tool_declaration])  # type: ignore
            
            max_retries = 8
            base_delay = 10.0
            
            for attempt in range(max_retries):
                try:
                    return model.generate_content(prompt)
                except exceptions.ResourceExhausted as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"⚠️ Quota exceeded (429). Retrying in {base_delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(base_delay)
                    # For Free Tier limits (15 RPM), waiting ~15-30s helps clear the burst window.
                    base_delay *= 1.5
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"⚠️ Quota exceeded string match (429). Retrying in {base_delay}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(base_delay)
                        base_delay *= 1.5
                    else:
                        raise e
            
        return _execute_with_timeout(_call_llm, timeout_seconds)

    def run(self, question: str) -> AgentResult:
        context = TaskContext(query=question)
        start_time = time.time()
        logger.info(f"🚀 Started Agent Execution (Max {self.max_iterations} loops, {self.timeout_seconds}s timeout)")
        
        pending_tool_name = ""
        pending_tool_args: Any = {}
        pending_tool_result: str = ""
        
        while context.iterations < self.max_iterations:
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                logger.warning(f"⏳ Timeout of {self.timeout_seconds}s reached. Forcing partial synthesis.")
                return self._force_partial_synthesis(context, "budget_exhausted_timeout")
                
            context.iterations += 1
            logger.info(f"🔄 Step {context.iterations}: {context.current_state.name}")
            
            if context.current_state == AgentState.PLANNING:
                history_str = "\n".join(context.reasoning_history) if context.reasoning_history else "None."
                prompt = (
                    f"{self.system_prompt_base}\n\n"
                    f"USER QUESTION: {context.query}\n\n"
                    f"PREVIOUS REASONING HISTORY:\n{history_str}\n\n"
                    f"{context.summarize_context_for_prompt()}\n\n"
                    f"What is your next action?"
                )
                
                try:
                    time_left = self.timeout_seconds - (time.time() - start_time)
                    resp = self._generate(prompt, timeout_seconds=time_left)
                except Exception as e:
                    logger.error(f"❌ Planning LLM Error: {str(e)}")
                    # If the LLM timeouts or crashes, we break the loop and force synthesis.
                    return self._force_partial_synthesis(context, f"budget_exhausted_error_{type(e).__name__}")
                
                tool_name: Optional[str] = None
                tool_args: Any = {}
                final_answer: Optional[str] = None
                raw_response_log: str = ""
                try:
                    if resp.parts:
                        part = resp.parts[0]
                        if getattr(part, 'function_call', None):
                            tool_name = part.function_call.name
                            tool_args = dict(part.function_call.args)
                            raw_response_log = f"FunctionCall: {tool_name}({tool_args})"
                        else:
                            final_answer = part.text
                            raw_response_log = final_answer
                except AttributeError:
                    final_answer = getattr(resp, 'text', "Error parsing response")
                    raw_response_log = final_answer
                
                self.tracer.log_step(context.iterations, context.current_state.name, prompt, raw_response_log, tool_name)
                
                if tool_name and self.tool_registry.has_tool(tool_name):
                    pending_tool_name = tool_name
                    pending_tool_args = tool_args
                    context.reasoning_history.append(f"Step {context.iterations}: LLM called '{tool_name}'")
                    context.current_state = AgentState.RETRIEVING
                elif final_answer:
                    context.final_answer = final_answer
                    context.current_state = AgentState.SYNTHESIZING
                else:
                    context.reasoning_history.append("System: Unexpected LLM output.")
                
            elif context.current_state == AgentState.RETRIEVING:
                logger.info(f"   -> Executing: {pending_tool_name} with args: {pending_tool_args}")
                is_error = False
                try:
                    time_left = self.timeout_seconds - (time.time() - start_time)
                    tool_time_budget = min(time_left, self.tool_timeout_seconds)
                    tool = self.tool_registry.get_tool(pending_tool_name)
                    pending_tool_result = str(_execute_with_timeout(tool.execute, tool_time_budget, pending_tool_args))
                except Exception as e:
                    logger.error(f"❌ Tool Execution Error: {str(e)}")
                    pending_tool_result = f"Tool '{pending_tool_name}' execution failed with exception: {str(e)}"
                    is_error = True
                    
                self.tracer.log_step(
                    step_number=context.iterations,
                    state=context.current_state.name,
                    input_text=json.dumps(pending_tool_args),
                    output_text=pending_tool_result,
                    tool_called=pending_tool_name,
                    is_error=is_error
                )
                
                if is_error:
                    # Bypass verification and feed the exception directly back to the Planner
                    context.reasoning_history.append(f"System: {pending_tool_result}")
                    context.current_state = AgentState.PLANNING
                else:
                    context.current_state = AgentState.EVALUATING
                context.retrieval_count += 1
                
            elif context.current_state == AgentState.EVALUATING:
                snippet_id = f"step_{context.iterations}_{pending_tool_name}"
                context.gathered_context[snippet_id] = pending_tool_result
                
                # 1. Deterministic Trap: Don't trust the LLM if the tool literally returned nothing
                if pending_tool_result and ("No semantic overlap found" in str(pending_tool_result) or "ERROR" in str(pending_tool_result)):
                    logger.warning(f"   -> Verification: Snippet {snippet_id} returned empty/error. Forcing REFORMULATE.")
                    verify_result = "REFORMULATE"
                    verify_prompt = "N/A (Deterministic Block)"
                    
                else:
                    logger.info(f"   -> Verification: Snippet {snippet_id} captured. Pinging LLM to evaluate if DONE.")
                    
                    # 2. The Actor-Critic Verification Prompt (Now enforcing Chain-of-Thought)
                    verify_prompt = (
                        f"USER QUESTION: {context.query}\n\n"
                        f"GATHERED CONTEXT SO FAR:\n{context.summarize_context_for_prompt()}\n\n"
                        f"Does the gathered context contain enough information to fully and accurately answer the user's question?\n"
                        f"First, write a 1-sentence reasoning explaining why the context is or isn't sufficient.\n"
                        f"Then, on a new line, reply with exactly one word: 'VERDICT: DONE' if yes, or 'VERDICT: REFORMULATE' if you need to search again."
                    )
                    
                    try:
                        time_left = self.timeout_seconds - (time.time() - start_time)
                        resp = self._generate(verify_prompt, timeout_seconds=time_left)
                        raw_verify_text = getattr(resp, 'text', "VERDICT: REFORMULATE").strip().upper()
                        if not raw_verify_text:
                            raw_verify_text = "VERDICT: REFORMULATE" if not hasattr(resp, 'parts') else (resp.parts[0].text.strip().upper() if resp.parts else "VERDICT: REFORMULATE")
                            
                        # Extract just the verdict from the CoT output
                        if "VERDICT: DONE" in raw_verify_text:
                            verify_result = "DONE"
                        else:
                            verify_result = "REFORMULATE"
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Verification timeout/error: {str(e)}. Forcing partial synthesis.")
                        return self._force_partial_synthesis(context, f"budget_exhausted_error_{type(e).__name__}")
                        
                self.tracer.log_step(context.iterations, context.current_state.name, verify_prompt, verify_result, "verify_context")
                
                if verify_result == "DONE":
                    context.reasoning_history.append("System: Context verified as sufficient. Synthesizing final answer.")
                    # Generate the final answer from gathered context via a synthesis LLM call
                    synthesis_prompt = (
                        f"You are answering the following question using ONLY the verified codebase context below.\n\n"
                        f"USER QUESTION: {context.query}\n\n"
                        f"{context.summarize_context_for_prompt()}\n\n"
                        f"Provide a complete, accurate answer with specific code references."
                    )
                    try:
                        time_left = self.timeout_seconds - (time.time() - start_time)
                        resp = self._generate(synthesis_prompt, timeout_seconds=time_left)
                        context.final_answer = getattr(resp, 'text', None)
                        if not context.final_answer and hasattr(resp, 'parts') and resp.parts:
                            context.final_answer = resp.parts[0].text
                    except Exception as e:
                        logger.warning(f"⚠️ Synthesis LLM error after DONE verdict: {str(e)}")
                        context.final_answer = None
                    context.current_state = AgentState.SYNTHESIZING
                else:
                    if context.retrieval_count >= self.max_reformulations:
                        logger.warning(f"⚠️ Reformulation bound reached ({self.max_reformulations} searches). Forcing synthesis.")
                        context.reasoning_history.append("System: Context deemed insufficient, but max reformulations reached. Synthesizing.")
                        context.current_state = AgentState.SYNTHESIZING
                    else:
                        logger.info("   -> Verification: Context insufficient. Returning to PLANNING to reformulate.")
                        context.reasoning_history.append(f"System: Tool {pending_tool_name} executed, but context is insufficient. Must reformulate search.")
                        context.current_state = AgentState.PLANNING
                
            elif context.current_state == AgentState.SYNTHESIZING:
                logger.info("✅ SYNTHESIZING complete.")
                self.tracer.log_step(
                    step_number=context.iterations,
                    state=context.current_state.name,
                    input_text="Final generation triggered.",
                    output_text=context.final_answer,
                    tool_called=None
                )
                return AgentResult(
                    answer=context.final_answer or "No answer synthesized",
                    iterations=context.iterations,
                    success=True,
                    termination_reason="success",
                    context_used=context.gathered_context
                )
                
        logger.warning(f"❌ Max iterations ({self.max_iterations}) reached. Forcing partial synthesis.")
        return self._force_partial_synthesis(context, "budget_exhausted_max_steps")
        
    def _force_partial_synthesis(self, context: TaskContext, reason: str) -> AgentResult:
        """Fallback handler to prevent infinite loops and timeout crashes."""
        logger.info("   -> Ping LLM to synthesize partial answer from gathered context.")
        prompt = (
            f"You have reached the maximum search depth or time limit for the question: '{context.query}'.\n"
            f"You must immediately provide the best partial answer you can using ONLY the following context. "
            f"Explicitly state what information you were unable to find.\n\n"
            f"{context.summarize_context_for_prompt()}"
        )
        
        try:
            # We enforce a hard 15 second timeout for the fallback synthesis
            resp = self._generate(prompt, timeout_seconds=15.0)
            partial_answer = getattr(resp, 'text', "Error generating partial answer") if not hasattr(resp, 'parts') else (resp.parts[0].text if resp.parts else "Empty response")
        except Exception as e:
            partial_answer = f"Failed to generate partial answer. Error: {str(e)}"
            
        self.tracer.log_step(
            step_number=context.iterations,
            state="FORCED_SYNTHESIS",
            input_text=prompt,
            output_text=partial_answer,
            tool_called=f"termination_reason: {reason}"
        )
            
        return AgentResult(
            answer=partial_answer,
            iterations=context.iterations,
            success=False,
            termination_reason=reason,
            context_used=context.gathered_context
        )
