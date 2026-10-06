# langgraph-tool-agent

[![CI](https://github.com/shu0819-sjy/langgraph-tool-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/shu0819-sjy/langgraph-tool-agent/actions/workflows/ci.yml)

A minimal **LangGraph** tool-calling agent that demonstrates **reliable tool-failure handling**:
`tool failure -> bounded retry (1x) -> graceful fallback`.

Built as a companion to my other projects ([llm-router](https://github.com/shu0819-sjy/llm-router)):
in production agent systems, most failures are *tool-call failures* — invalid arguments,
transient upstream errors, silent timeouts. Handling them well is what makes an agent usable.

## What it shows

- A `StateGraph` with an `agent <-> tools` loop (hand-rolled, no prebuilt helper)
- 3 purely local tools: `calculator` (AST-whitelisted safe eval), `current_time`, `mock_weather`
- **Failure paths, on purpose:**
  - `mock_weather`'s first call always fails with a simulated transient error ->
    the agent retries once and succeeds
  - asking for an unknown city fails on every attempt -> the tool node returns a
    graceful fallback message instead of crashing the graph

## Quickstart

Requires Python 3.11 or newer. Use a virtual environment so the LangGraph
dependencies stay isolated from other projects.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python agent.py
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python agent.py
```

Edit `.env` and set `OPENAI_API_KEY` before running the live demo. The optional
`OPENAI_BASE_URL` selects an OpenAI-compatible endpoint and `OPENAI_MODEL`
selects the model. `.env` is local-only and must not be committed. Importing
`agent` and running the tests do not initialize an LLM or require a real key.

Expected output (abridged):

```
=== USER: Use the tools: what is the weather in Tokyo, and what is 12*(3+4)?
    -> tool call: mock_weather {'city': 'Tokyo'}
    [tool mock_weather] attempt 1 failed: simulated transient upstream error
    -> tool call: mock_weather {'city': 'Tokyo'}          # retry succeeds
    -> tool call: calculator {'expression': '12*(3+4)'}

=== USER: Use the tools: what is the weather in Atlantis?
    [tool mock_weather] attempt 1 failed: unknown city: Atlantis
    [tool mock_weather] attempt 2 failed: unknown city: Atlantis
    -> fallback: (mock_weather is currently unavailable; please try again later.)
```

## Tool failure handling

The core is `run_tool_with_fallback()` in `agent.py`:

1. try the tool
2. on any exception, retry once (bounded retry)
3. if the retry also fails, return a fallback message as the tool result — the graph
   keeps running and the LLM can tell the user something useful instead of the
   whole invocation crashing

## Offline tests

The tests exercise the calculator, local weather failure injection and retry
fallback without network access or model downloads. Run them on every platform:

```bash
python -m pytest
python -m ruff check agent.py tests
```

The test suite uses a clearly fake key only for compatibility construction. It
never sends a request. Importing `agent` without `OPENAI_API_KEY` is supported
for local-tool usage and offline tests.

## License

MIT — Copyright (c) 2026 langgraph-tool-agent contributors
