"""
Pattern 3: Plan-and-Execute

The model writes a full multi-step plan up front, then executes each
step in order. Unlike ReAct, it does not re-reason from scratch after
every single action. It only re-plans if a step reveals something
that changes the rest of the plan (not implemented here, kept simple
on purpose, see the README for how to add it).

When to use: multi-step workflows where the steps are largely known
in advance and efficiency, or a reviewable plan, matters.

Run:
    python main.py
"""

import os
import sys
from typing import List, TypedDict
from dotenv import load_dotenv



sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
load_dotenv()

model = init_chat_model("google_genai:gemini-2.5-flash", temperature=0)


class PlanState(TypedDict):
    task: str
    steps: List[str]
    current_step: int
    results: List[str]


def planner(state: PlanState) -> dict:
    prompt = (
        "Break this task into 2 to 4 short, concrete steps, one per line, "
        f"no numbering, no extra text:\n\n{state['task']}"
    )
    response = model.invoke(prompt)
    steps = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
    print("Plan:")
    for i, s in enumerate(steps):
        print(f"  {i + 1}. {s}")
    return {"steps": steps, "current_step": 0, "results": []}


def executor(state: PlanState) -> dict:
    step = state["steps"][state["current_step"]]
    print(f"\nExecuting step {state['current_step'] + 1}: {step}")

    # A production version would let the model choose a tool per step.
    # Kept simple here: the model answers the step directly, with prior
    # results passed in as context.
    context = "\n".join(state["results"])
    response = model.invoke(f"Prior results:\n{context}\n\nDo this step: {step}")

    return {
        "results": state["results"] + [response.content],
        "current_step": state["current_step"] + 1,
    }


def should_continue(state: PlanState) -> str:
    return "executor" if state["current_step"] < len(state["steps"]) else END


graph = StateGraph(PlanState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue)
app = graph.compile()


def main():
    task = (
        "Explain what overfitting is, give one real-world analogy, "
        "and suggest two ways to prevent it."
    )
    final_state = app.invoke({"task": task, "steps": [], "current_step": 0, "results": []})

    print("\n--- Final combined result ---")
    print("\n\n".join(final_state["results"]))


if __name__ == "__main__":
    main()
