# -*- coding: utf-8 -*-
"""
langgraph-tool-agent — minimal LangGraph tool-calling agent with reliable
tool-failure handling (failure -> bounded retry -> graceful fallback).

Quickstart:
    pip install -r requirements.txt
    copy .env.example to .env and fill in OPENAI_API_KEY
    py -3 agent.py
"""

import ast
import operator
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

# ---------- 0. 环境变量：密钥只从 .env / 环境读取，绝不写死在代码里 ----------
load_dotenv()
BASE_URL = os.getenv("OPENAI_BASE_URL") or None  # 可选：任意 OpenAI 兼容端点
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------- 1. 三个纯本地工具，不需要任何外部服务 ----------

# 计算器：用 ast 白名单求值，杜绝 eval 注入
_CALC_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")

def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression like '12*(3+4)'. Safe: AST whitelist only."""
    return str(_safe_eval(ast.parse(expression, mode="eval").body))

def current_time(_: str = "") -> str:
    """Return the current local time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

_WEATHER = {"Tokyo": "Sunny, 24C", "Singapore": "Rainy, 29C", "London": "Cloudy, 15C"}

# 失败注入计数器：模拟「瞬时故障」——mock_weather 的第一次调用固定失败，
# 重试成功；这样每次运行都能看到完整的 failure -> retry -> success 链路。
_weather_calls = []

def mock_weather(city: str) -> str:
    """Look up weather for a demo city. First call simulates a transient failure;
    unknown cities always fail (demonstrates the graceful fallback path)."""
    _weather_calls.append(city)
    if len(_weather_calls) == 1:
        raise RuntimeError("simulated transient upstream error")
    if city not in _WEATHER:
        raise ValueError(f"unknown city: {city}")
    return _WEATHER[city]

TOOLS = {"calculator": calculator, "current_time": current_time, "mock_weather": mock_weather}

# ---------- 2. 带兜底的工具执行节点：failure -> retry(1) -> fallback ----------

def run_tool_with_fallback(name: str, args: dict) -> str:
    """执行一个工具；失败后重试 1 次；仍失败则返回兜底文案而不是抛异常，
    保证 agent 图不中断 —— 这是本仓库要演示的核心行为。"""
    for attempt in (1, 2):
        try:
            return TOOLS[name](**args)
        except Exception as exc:  # noqa: BLE001 — 演示用：任何工具异常都走同一处理路径
            print(f"    [tool {name}] attempt {attempt} failed: {exc}")
    return f"({name} is currently unavailable; please try again later.)"

def call_tools(state: MessagesState):
    """手动执行上一条 AI 消息里的所有 tool_calls，包成 ToolMessage 返回。"""
    last: AIMessage = state["messages"][-1]
    out = []
    for tc in last.tool_calls:
        print(f"    -> tool call: {tc['name']} {tc['args']}")
        result = run_tool_with_fallback(tc["name"], tc["args"] or {})
        out.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return {"messages": out}

# ---------- 3. 模型节点 + 路由 ----------

llm = ChatOpenAI(model=MODEL, temperature=0, base_url=BASE_URL).bind_tools(
    list(TOOLS.values())
)

def call_model(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

def route(state: MessagesState):
    return "tools" if state["messages"][-1].tool_calls else END

# ---------- 4. 组图：agent <-> tools 循环 ----------

graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()

# ---------- 5. 两个演示：瞬时故障重试 / 未知城市兜底 ----------

DEMOS = [
    "Use the tools: what is the weather in Tokyo, and what is 12*(3+4)?",
    "Use the tools: what is the weather in Atlantis?",
]

if __name__ == "__main__":
    # Windows 控制台默认 GBK：模型输出可能包含 emoji 等字符，强制 UTF-8 输出避免崩编码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for q in DEMOS:
        print(f"\n=== USER: {q}")
        result = app.invoke({"messages": [("user", q)]})
        for m in result["messages"]:
            tag = type(m).__name__
            text = getattr(m, "content", "") or (f"tool_calls={getattr(m, 'tool_calls', None)}")
            print(f"[{tag}] {text}")
