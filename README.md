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

```bash
pip install -r requirements.txt
copy .env.example .env    # then fill in OPENAI_API_KEY (any OpenAI-compatible base URL works too)
py -3 agent.py            # or: python agent.py
```

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

## License

MIT — Copyright (c) 2026 langgraph-tool-agent contributors
