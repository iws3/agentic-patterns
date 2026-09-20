"""
Pattern 5: Multi-Agent (Supervisor / Orchestrator-Worker)

A supervisor model routes each request to one of several specialized
sub-agents, each with its own narrow toolset and prompt, instead of
one agent trying to juggle every tool and every domain at once.

When to use: your toolset or task variety has grown large enough that
one system prompt describing everything gets unwieldy, or you notice
the model picking the wrong tool because too many options compete for
its attention.

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

model = init_chat_model("gpt-4o-mini", temperature=0)

math_agent = create_react_agent(
    model, [calculator], prompt="You handle math only. Use the calculator tool for any arithmetic."
)
research_agent = create_react_agent(
    model, [web_search, get_weather], prompt="You handle research and weather questions only."
)

AGENTS = {"math": math_agent, "research": research_agent}


def supervisor_route(task: str) -> str:
    prompt = (
        "Which specialist should handle this request: math or research? "
        f"Reply with exactly one word.\n\nRequest: {task}"
    )
    decision = model.invoke(prompt).content.strip().lower()
    return decision if decision in AGENTS else "research"


def main():
    tasks = [
        "What is 128 divided by 4, times 6?",
        "What's the weather like in Nairobi right now?",
    ]

    for task in tasks:
        route = supervisor_route(task)
        print(f"\nTask: {task}\nRouted to: {route}")

        agent = AGENTS[route]
        result = agent.invoke({"messages": [{"role": "user", "content": task}]})
        print(f"Answer: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
