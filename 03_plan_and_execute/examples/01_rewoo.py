"""
Sample 1: ReWOO-style Plan-and-Execute

One LLM call writes the whole plan. Each step in the plan is a tool call
whose output is stored in a variable (#E1, #E2, ...). Later steps can use
earlier results by variable name. A worker runs the tool calls with plain
Python, with no LLM in between. One final LLM call (the solver) reads the
plan and the evidence and writes the answer.

LLM calls for a whole run: 2 (planner + solver).

Run:
    python main.py
"""

import re
from datetime import date
from typing import Dict, List, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph

load_dotenv()

model = init_chat_model("google_genai:gemini-2.5-flash", temperature=0)


# ---------------------------------------------------------------------------
# Tools: plain Python functions. No LLM involved when these run.
# ---------------------------------------------------------------------------
ENROLLMENT = {"ANN": 24, "CNN": 21, "Transfer Learning": 19}
DEADLINES = {"ANN": "2026-09-01", "CNN": "2026-10-30", "Transfer Learning": "2026-11-20"}


def count_students(arg: str) -> str:
    return str(ENROLLMENT.get(arg.strip(), 0))


def get_deadline(arg: str) -> str:
    return DEADLINES.get(arg.strip(), "unknown")


def days_until(arg: str) -> str:
    try:
        target = date.fromisoformat(arg.strip())
    except ValueError:
        return f"error: '{arg}' is not a YYYY-MM-DD date"
    return str((target - date.today()).days)


TOOLS = {
    "CountStudents": count_students,
    "GetDeadline": get_deadline,
    "DaysUntil": days_until,
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class RewooState(TypedDict):
    task: str
    plan: str
    steps: List[tuple]          # (variable, tool name, argument)
    results: Dict[str, str]     # {"#E1": "21", ...}
    current: int
    answer: str


PLANNER_PROMPT = """You can use these tools:
CountStudents[module] - number of students enrolled in a module
GetDeadline[module] - submission deadline of a module, as YYYY-MM-DD
DaysUntil[date] - number of days from today until a YYYY-MM-DD date

Write a plan to answer the question. For every step write exactly two lines:
Plan: <short reason>
#E<n> = <Tool>[<argument>]

An argument may use an earlier result, for example #E2.
Write the plan only, no extra text.

Question: {task}"""

STEP_PATTERN = re.compile(r"(#E\d+)\s*=\s*(\w+)\[(.*?)\]")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def planner(state: RewooState) -> dict:
    plan = model.invoke(PLANNER_PROMPT.format(task=state["task"])).content.strip()
    steps = STEP_PATTERN.findall(plan)
    print("Plan written by the LLM:")
    print(plan)
    return {"plan": plan, "steps": steps, "results": {}, "current": 0}


def worker(state: RewooState) -> dict:
    variable, tool_name, arg = state["steps"][state["current"]]

    # Replace variables like #E1 with their stored results.
    # Longest names first, so #E10 is not damaged by replacing #E1.
    for name in sorted(state["results"], key=len, reverse=True):
        arg = arg.replace(name, state["results"][name])

    tool = TOOLS.get(tool_name)
    result = tool(arg) if tool else f"error: unknown tool '{tool_name}'"
    print(f"\nWorker: {variable} = {tool_name}[{arg}] -> {result}")

    return {
        "results": {**state["results"], variable: result},
        "current": state["current"] + 1,
    }


def solver(state: RewooState) -> dict:
    evidence = "\n".join(f"{var} = {value}" for var, value in state["results"].items())
    prompt = (
        f"Question: {state['task']}\n\nPlan:\n{state['plan']}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Answer the question using only the evidence above."
    )
    return {"answer": model.invoke(prompt).content.strip()}


def after_plan(state: RewooState) -> str:
    return "worker" if state["steps"] else "solver"


def after_worker(state: RewooState) -> str:
    return "worker" if state["current"] < len(state["steps"]) else "solver"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
graph = StateGraph(RewooState)
graph.add_node("planner", planner)
graph.add_node("worker", worker)
graph.add_node("solver", solver)
graph.set_entry_point("planner")
graph.add_conditional_edges("planner", after_plan)
graph.add_conditional_edges("worker", after_worker)
graph.add_edge("solver", END)
app = graph.compile()


def main():
    task = "How many students are in the CNN module, and how many days until its deadline?"
    final_state = app.invoke(
        {"task": task, "plan": "", "steps": [], "results": {}, "current": 0, "answer": ""}
    )
    print("\n--- Final answer ---")
    print(final_state["answer"])


if __name__ == "__main__":
    main()