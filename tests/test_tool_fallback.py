# -*- coding: utf-8 -*-
"""Teaching tests: tool failure -> bounded retry -> graceful fallback.

These tests never call a real LLM or network. They import ``agent`` with a
dummy ``OPENAI_API_KEY`` so module-level ChatOpenAI construction succeeds,
then exercise only the local tool helpers — the core lesson of this repo.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Importing agent.py builds ChatOpenAI at module scope; a dummy key is enough
# for construction and keeps CI offline.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")

import agent  # noqa: E402  — after env seed


@pytest.fixture(autouse=True)
def _reset_weather_calls():
    """mock_weather uses a module-level call log to inject the first failure."""
    agent._weather_calls.clear()
    yield
    agent._weather_calls.clear()


# ---------- pure local tools (readable teaching asserts) ----------


def test_calculator_safe_eval():
    assert agent.calculator("12*(3+4)") == "84"
    assert agent.calculator("2**10") == "1024"
    assert agent.calculator("-3+5") == "2"


def test_calculator_rejects_unsafe_expression():
    with pytest.raises(ValueError, match="unsupported"):
        agent.calculator("__import__('os').system('echo hi')")


def test_current_time_format():
    text = agent.current_time()
    # YYYY-MM-DD HH:MM:SS
    assert len(text) == 19
    assert text[4] == "-" and text[10] == " "


# ---------- core path: failure -> retry -> success / fallback ----------


def test_transient_failure_then_retry_success():
    """First mock_weather call always raises; retry must succeed for Tokyo."""
    result = agent.run_tool_with_fallback("mock_weather", {"city": "Tokyo"})
    assert result == "Sunny, 24C"
    assert agent._weather_calls == ["Tokyo", "Tokyo"]  # failed once, then ok


def test_permanent_failure_returns_fallback_not_raise():
    """Unknown city fails every attempt → graceful fallback string, no raise."""
    result = agent.run_tool_with_fallback("mock_weather", {"city": "Atlantis"})
    assert "unavailable" in result
    assert "mock_weather" in result
    assert agent._weather_calls == ["Atlantis", "Atlantis"]


def test_fallback_keeps_graph_contract_as_string():
    """Tool node must always return a string so ToolMessage stays valid."""
    out = agent.run_tool_with_fallback("mock_weather", {"city": "Nowhere"})
    assert isinstance(out, str)
    assert out.startswith("(")


def test_calculator_via_fallback_wrapper_no_retry_needed():
    """Happy path: success on first attempt, no fallback text."""
    assert agent.run_tool_with_fallback("calculator", {"expression": "1+1"}) == "2"


def test_weather_stateCanBeResetBetweenIndependentRuns():
    """验证独立演示运行可重置瞬时故障状态，避免跨运行共享重试计数。"""
    agent.reset_weather_state()
    with pytest.raises(RuntimeError, match="simulated transient"):
        agent.mock_weather("Tokyo")
    agent.reset_weather_state()
    with pytest.raises(RuntimeError, match="simulated transient"):
        agent.mock_weather("Tokyo")


def test_importDoesNotRequireApiKey():
    """验证仅导入本地工具不会初始化 LLM，也不会要求真实 API key。"""
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("OPENAI_BASE_URL", None)
    environment.pop("OPENAI_MODEL", None)
    result = subprocess.run(
        [sys.executable, "-c", "import agent; print('import-ok')"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import-ok"


def test_unknown_tool_name_falls_back():
    """Missing tool key → KeyError on both attempts → fallback string."""
    out = agent.run_tool_with_fallback("no_such_tool", {})
    assert "unavailable" in out
