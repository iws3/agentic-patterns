"""
Pattern 2: ReAct (Reason, Act, Observe, repeat)

The model repeatedly reasons about what to do next, takes an action
(a tool call), observes the result, and decides whether it is done
or needs another action. LangGraph's create_react_agent builds this
loop for you. You just supply the model and the tools.

When to use: open-ended requests where you cannot predict in advance
which tools, if any, a given message will need.

Run:
    python main.py
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from shared.tools import calculator, get_weather, web_search

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant. Only call a tool when the
request genuinely needs it. For general knowledge or conversation,
answer directly without calling anything."""


def main():
    model = init_chat_model(model="google_genai:gemini-2.5-flash", temperature=0)
    agent = create_react_agent(
        model, [get_weather, calculator, web_search], prompt=SYSTEM_PROMPT
    )

    # This single request needs two tools, in an order the model figures out itself.
    user_message = "What's the weather in Lagos, and what's 15% of 240?"
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"recursion_limit": 10},
    )

    for msg in result["messages"]:
        role = getattr(msg, "type", "unknown")
        name = getattr(msg, "name", None)
        label = f"{role} ({name})" if name else role
        print(f"[{label}] {msg.content}")


if __name__ == "__main__":
    main()
