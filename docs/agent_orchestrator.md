# Agent Orchestrator Architecture

The `AgentOrchestrator` is a mathematically rigid, deterministic state machine that manages the interaction between the LLM and our system tools. It is designed to prevent infinite hallucination loops, deadlocks, and premature satisfaction.

## 1. The State Machine Loop

The Orchestrator operates on a 4-node state machine:

1. **`PLANNING`**: The LLM evaluates the user's query, the accumulated reasoning history, and any context gathered so far. It makes a decision to either call a tool via Native Function Calling or synthesize an answer. If a tool is requested, the system transitions to `RETRIEVING`.
2. **`RETRIEVING`**: The Orchestrator intercepts the requested tool and executes it in an isolated, wall-clock bounded thread. The tool's output (or a caught Python Exception) is collected, and the system transitions to `EVALUATING` (unless an error occurred, in which case it routes back to `PLANNING` so the LLM can see the stack trace and self-correct).
3. **`EVALUATING`** *(The Actor-Critic)*: A decoupled LLM call evaluates the tool's output. Using Chain-of-Thought prompting, it explains why the context is sufficient or lacking, then outputs exactly `VERDICT: DONE` (transitions to `PLANNING` to synthesize) or `VERDICT: REFORMULATE` (loops back to `PLANNING` for another search).
4. **`SYNTHESIZING`**: Once verified, or if the system hits a budget limit, the agent provides the final text answer based purely on the verified context.

### Guardrails
* **Wall-Clock Budget**: All external operations are wrapped in a hard timeout executor.
* **Deterministic Fallback (`_force_partial_synthesis`)**: If the agent reaches `max_iterations`, times out, or violently crashes, the orchestrator safely intercepts and asks the LLM to output a partial answer using whatever context it gathered prior to the fault.

---

## 2. Telemetry & Trace Schema

We maintain 100% forensic visibility into the LLM's thought process via `traces.db`. This SQLite database is updated in real-time, meaning we don't lose logs if a process `SIGKILL`s.

The `traces` table schema:

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER PRIMARY KEY` | Auto-incrementing row ID. |
| `run_id` | `TEXT` | A unique UUID representing a single `orchestrator.run()` execution. |
| `step_number` | `INTEGER` | The current loop iteration (`context.iterations`). |
| `state` | `TEXT` | The current state (`PLANNING`, `RETRIEVING`, `EVALUATING`, etc). |
| `tool_called` | `TEXT` | The name of the tool executed (e.g., `search_codebase`). |
| `input` | `TEXT` | The raw injected Prompt to the LLM, or the JSON args to the Tool. |
| `output` | `TEXT` | The raw text/function-call from the LLM, or the string output of the Tool. |
| `timestamp` | `REAL` | Unix epoch time. |

---

## 3. Adding a New Tool

Tools are entirely decoupled from the Orchestrator loop via the `ToolRegistry`. To add a new capability (e.g. `sandbox`, `read_file`, `git`), you only need to write the tool class and register it.

### Step 1: Create the Tool Class
Create a new file in `src/agent/tools/` and inherit from `BaseTool`.

```python
from typing import Dict, Any
from agent.tools.base import BaseTool

class ReadFileTool(BaseTool):
    @property
    def name(self) -> str:
        return "read_file"
        
    @property
    def description(self) -> str:
        return "Reads the absolute contents of a file on disk."

    def execute(self, args: Dict[str, Any]) -> str:
        # 1. Parse arguments safely
        filepath = args.get("filepath")
        if not filepath:
            return "Error: Missing filepath."
            
        # 2. Execute logic and return a RAW STRING
        try:
            with open(filepath, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def get_function_declaration(self) -> Dict[str, Any]:
        # Return the exact Gemini/OpenAI Native Function Calling JSON schema
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "filepath": {
                        "type": "STRING",
                        "description": "Absolute path to the file."
                    }
                },
                "required": ["filepath"]
            }
        }
```

### Step 2: Register the Tool

In `main.py` or wherever you initialize your Orchestrator, register the tool:

```python
from agent.orchestrator import AgentOrchestrator
from agent.tools.registry import ToolRegistry
from agent.tools.search_codebase import SearchCodebaseTool
from agent.tools.read_file import ReadFileTool

registry = ToolRegistry()
registry.register(SearchCodebaseTool())
registry.register(ReadFileTool())

agent = AgentOrchestrator(tool_registry=registry)
result = agent.run("What is in main.py?")
```

The Orchestrator will automatically inject your tool's schema into the `PLANNING` prompt and seamlessly route execution to your `execute()` method without any modifications to the core engine.
