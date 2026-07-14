import pytest
from unittest.mock import patch, MagicMock

from agent.orchestrator import AgentOrchestrator
from agent.state import AgentState
from tests.conftest import MockLLMClient, MockResponse

class MockFunctionCall:
    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args

def test_orchestrator_normal_completion(mock_llm: MockLLMClient):
    """Test a perfect 1-hop search-and-answer flow."""
    # 1. Plan -> Search
    step1 = MockResponse(function_call=MockFunctionCall("search_codebase", {"query": "test"}))
    # 2. Evaluate -> DONE
    step2 = MockResponse(text="The context is perfect. VERDICT: DONE")
    # 3. Synthesize -> Answer
    step3 = MockResponse(text="Here is the final answer.")
    
    mock_llm.set_responses([step1, step2, step3])
    
    orchestrator = AgentOrchestrator(llm_client=mock_llm)
    
    # Mock the DB tool so we don't need a real Postgres connection
    with patch.object(orchestrator.tool_registry.get_tool('search_codebase'), 'execute', return_value="def test(): pass"):
        result = orchestrator.run("How does test work?")
        
        assert result.success is True
        assert result.termination_reason == "success"
        assert result.answer == "Here is the final answer."
        assert result.iterations == 5 # PLAN -> RETRIEVE -> EVALUATE -> PLAN -> SYNTHESIZE

def test_orchestrator_budget_exhaustion_max_steps(mock_llm: MockLLMClient):
    """Test what happens when the agent hallucinates endlessly and hits max_steps."""
    # We hit max_iterations=3. 
    # Loop 1 consumes responses[0]. Loop 2 consumes responses[1]. Loop 3 consumes responses[2].
    # Then it breaks the loop and the forced synthesis consumes responses[3].
    responses = [MockResponse(function_call=MockFunctionCall("invalid_tool", {})) for _ in range(3)]
    responses.append(MockResponse(text="Forced Partial Answer")) # For the fallback synthesis
    
    mock_llm.set_responses(responses)
    orchestrator = AgentOrchestrator(llm_client=mock_llm)
    orchestrator.max_iterations = 3 # Lower budget for test
    
    result = orchestrator.run("Endless loop")
    
    assert result.success is False
    assert result.termination_reason == "budget_exhausted_max_steps"
    assert result.answer == "Forced Partial Answer"
    assert result.iterations == 3

def test_orchestrator_tool_exception_mid_run(mock_llm: MockLLMClient):
    """Test that a python exception inside a tool routes back to PLANNING, not EVALUATING."""
    # 1. Plan -> Search
    step1 = MockResponse(function_call=MockFunctionCall("search_codebase", {"query": "test"}))
    # 2. Plan (Self-Correction after error) -> Answer
    step2 = MockResponse(text="Final answer without tool context.")
    
    mock_llm.set_responses([step1, step2])
    orchestrator = AgentOrchestrator(llm_client=mock_llm)
    
    # Mock the tool to crash violently
    def _crash(*args, **kwargs):
        raise ValueError("Database connection dropped!")
        
    with patch.object(orchestrator.tool_registry.get_tool('search_codebase'), 'execute', side_effect=_crash):
        result = orchestrator.run("Do something")
        
        # It should survive the crash, self-correct, and still finish successfully
        assert result.success is True
        assert result.termination_reason == "success"
        assert result.answer == "Final answer without tool context."

def test_orchestrator_timeout_wall_clock(mock_llm: MockLLMClient):
    """Test that the wall-clock timeout successfully interrupts a hanging loop."""
    import time
    
    orchestrator = AgentOrchestrator(llm_client=mock_llm)
    
    # Better way: mock time.time to simulate time passing during the while loop evaluation
    with patch('time.time', side_effect=[100.0, 100.0, 200.0, 200.0, 200.0]):
        # Start time = 100
        # Loop 1 elapsed = 100 - 100 = 0
        # Next loop elapsed = 200 - 100 = 100 > timeout!
        orchestrator.timeout_seconds = 50
        
        # Responses: Step 1 (Fallback): Partial answer. The loop timeout prevents the first LLM call.
        mock_llm.set_responses([
            MockResponse(text="Forced Partial Answer")
        ])
        
        result = orchestrator.run("Timeout test")
        
        assert result.success is False
        assert result.termination_reason == "budget_exhausted_error_TimeoutError"
        assert result.answer == "Forced Partial Answer"
