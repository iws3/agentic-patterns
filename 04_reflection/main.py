"""
Pattern 4: Reflection (self-critique)

The model drafts an answer, then critiques its own draft, then
revises if the critique found problems. Loops until it passes or a
retry cap is hit, similar in spirit to recursion_limit in ReAct.

When to use: quality-sensitive generation where first drafts are
known to have rough edges (long-form writing, prompts for other
tools, code).

Run:
    python main.py
"""

import os
import sys
from typing import TypedDict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph

model = init_chat_model("gpt-4o-mini", temperature=0)

MAX_REVISIONS = 3


class ReflectState(TypedDict):
    task: str
    draft: str
    critique: str
    needs_revision: bool
    revision_count: int


def draft_node(state: ReflectState) -> dict:
    response = model.invoke(f"Write a short answer to: {state['task']}")
    return {"draft": response.content}


def critique_node(state: ReflectState) -> dict:
    prompt = (
        "Critique this draft for clarity, correctness, and completeness. "
        "If it is genuinely good, reply with exactly APPROVED. "
        f"Otherwise, explain what to fix.\n\nTask: {state['task']}\n\nDraft: {state['draft']}"
    )
    response = model.invoke(prompt)
    approved = response.content.strip().upper().startswith("APPROVED")
    print(f"\nCritique (revision {state['revision_count']}): {response.content}")
    return {"critique": response.content, "needs_revision": not approved}


def revise_node(state: ReflectState) -> dict:
    prompt = (
        "Revise this draft based on the feedback.\n\n"
        f"Original task: {state['task']}\n\nDraft: {state['draft']}\n\nFeedback: {state['critique']}"
    )
    response = model.invoke(prompt)
    return {"draft": response.content, "revision_count": state["revision_count"] + 1}


def route(state: ReflectState) -> str:
    if not state["needs_revision"] or state["revision_count"] >= MAX_REVISIONS:
        return END
    return "revise"


graph = StateGraph(ReflectState)
graph.add_node("draft", draft_node)
graph.add_node("critique", critique_node)
graph.add_node("revise", revise_node)
graph.set_entry_point("draft")
graph.add_edge("draft", "critique")
graph.add_conditional_edges("critique", route)
graph.add_edge("revise", "critique")
app = graph.compile()


def main():
    task = "Explain the difference between precision and recall to a beginner."
    final_state = app.invoke(
        {"task": task, "draft": "", "critique": "", "needs_revision": True, "revision_count": 0}
    )
    print("\n--- Final answer ---")
    print(final_state["draft"])


if __name__ == "__main__":
    main()
