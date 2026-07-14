# ADR 5: Orchestrator State Machine and Tool Calling Architecture

**Date:** 2026-07-14
**Status:** Accepted
**Author(s):** Divyansh

## Context
During Phase 2 (Raw Hybrid Retrieval), we identified the "Synthesis Problem"—our vector database could pull the exact 10 code chunks necessary to answer complex, multi-hop architectural questions, but the zero-shot LLM would fail to connect the dots. To solve this, we moved to an Agentic Orchestrator loop (Phase 3). 
However, LLMs hallucinate frequently when querying tools, often entering infinite loops or satisficing (prematurely deciding they have the answer when they only have a tangential chunk). We need a rigid, mathematically proven control flow to safely orchestrate an LLM's interactions with our tooling.

## Decision
We implemented a rigid 4-node State Machine (`PLANNING` $\rightarrow$ `RETRIEVING` $\rightarrow$ `EVALUATING` $\rightarrow$ `SYNTHESIZING`) with the following architectural choices:

1. **State Machine Control Flow**: The agent cannot execute arbitrary Python. It must output Native Function Calling JSON which our `AgentOrchestrator` parses to transition states.
2. **Actor-Critic Verification**: We decoupled the decision to search from the decision to stop. The `PLANNING` prompt is optimized solely for invoking tools. After a tool returns data, an isolated `EVALUATING` prompt (using Chain-of-Thought) forces the LLM to write a 1-sentence rationale before outputting exactly `VERDICT: DONE` or `VERDICT: REFORMULATE`.
3. **Wall-Clock Budgeting**: To prevent network requests or sandbox executions from hanging the agent, we wrapped all tool executions in a background `ThreadPoolExecutor` (soon to be upgraded to `subprocess` with `SIGTERM`). Timeouts are enforced at the strict wall-clock level rather than relying on max iteration counts.
4. **Native Tool Calling**: We abandoned custom XML-parsing in favor of Gemini's Native Function Calling APIs to guarantee schema compliance.

## Alternatives Considered
- **Custom XML/Regex Parsing:** Rejected. Initially considered for full control, but LLMs frequently missed closing tags or formatted the XML incorrectly, requiring complex regex recovery blocks. Native Function Calling provides guaranteed JSON schema validation.
- **Single-Prompt ReAct (Reason+Act):** Rejected. Trying to prompt the LLM to both parse new context and decide if it was "done" led to the Satisficing Bug (the LLM was lazy and yelled `DONE` the moment it saw a similar variable name). Splitting this into an isolated Actor-Critic `EVALUATING` state forces honesty.
- **Soft Iteration Limits:** Rejected. Relying purely on `max_iterations=6` is dangerous because a single tool call could hang the process infinitely. We require a hard wall-clock timeout wrapper on all I/O.

## Consequences
- **Positive:**
  - **Unbreakable Guardrails:** The deterministic state machine prevents the LLM from hallucinating unauthorized actions. If a tool crashes violently, the Python Exception is caught, logged, and fed directly back into `PLANNING` so the LLM can read the stack trace and self-correct.
  - **Extensibility:** The loop is now entirely decoupled from specific tools. New tools can be injected into the `ToolRegistry` and the state machine will route them flawlessly.
- **Negative:**
  - **Latency and Cost:** Splitting planning and verification into two separate LLM calls per loop roughly doubles token consumption and adds significant API latency.
  - **Prompt Sensitivity:** The `EVALUATING` prompt requires aggressive Chain-of-Thought to prevent premature termination, which requires careful prompt-engineering to maintain.
