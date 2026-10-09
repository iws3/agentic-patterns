"""
Sample 2: BabyAGI-style task queue

There is no fixed plan. The agent keeps a queue of tasks and repeats a
cycle: execute the top task, create new tasks from the result, then
reorder the queue by priority. Because the original design has no
built-in notion of "done", this sample adds a MAX_ITERATIONS stop.

Three roles, three prompts:
    execute     -> completes the top task
    create      -> adds new tasks based on the last result
    prioritize  -> reorders the queue

Run:
    python main.py
"""

from typing import Dict, List, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph

load_dotenv()

model = init_chat_model("google_genai:gemini-2.5-flash", temperature=0)

MAX_ITERATIONS = 5


class BabyState(TypedDict):
    objective: str
    tasks: List[str]              # the queue, first item runs next
    done: List[Dict[str, str]]    # [{"task": ..., "result": ...}, ...]
    iteration: int


def clean_lines(text: str) -> List[str]:
    """Split model output into clean one-per-line items."""
    lines = []
    for line in text.strip().split("\n"):
        line = line.strip().lstrip("-*0123456789. )").strip()
        if line:
            lines.append(line)
    return lines


def execute(state: BabyState) -> dict:
    task = state["tasks"][0]
    recent = "\n".join(f"- {d['task']}: {d['result']}" for d in state["done"][-3:])
    prompt = (
        f"Objective: {state['objective']}\n"
        f"Recent results:\n{recent or 'none yet'}\n\n"
        f"Complete this task in at most 4 sentences: {task}"
    )
    result = model.invoke(prompt).content.strip()
    print(f"\n[Iteration {state['iteration'] + 1}] Executed: {task}")
    print(f"Result: {result}")
    return {
        "tasks": state["tasks"][1:],
        "done": state["done"] + [{"task": task, "result": result}],
        "iteration": state["iteration"] + 1,
    }


def create(state: BabyState) -> dict:
    last = state["done"][-1]
    known = [d["task"] for d in state["done"]] + state["tasks"]
    prompt = (
        f"Objective: {state['objective']}\n"
        f"Last completed task: {last['task']}\n"
        f"Its result: {last['result']}\n"
        f"Existing and completed tasks:\n" + "\n".join(f"- {t}" for t in known) + "\n\n"
        "List up to 2 NEW tasks still needed to reach the objective, one per line, "
        "no numbering. Do not repeat any existing task. "
        "If nothing more is needed, write NONE."
    )
    new_tasks = clean_lines(model.invoke(prompt).content)
    new_tasks = [t for t in new_tasks if t.upper() != "NONE"]

    # Drop duplicates, ignoring case
    seen = {t.lower() for t in known}
    fresh = []
    for t in new_tasks:
        if t.lower() not in seen:
            fresh.append(t)
            seen.add(t.lower())

    if fresh:
        print("New tasks:", fresh)
    return {"tasks": state["tasks"] + fresh}


def prioritize(state: BabyState) -> dict:
    if len(state["tasks"]) < 2:
        return {}
    prompt = (
        f"Objective: {state['objective']}\n"
        "Reorder these tasks, most important first. Return exactly the same "
        "tasks, one per line, no numbering, no extra text:\n"
        + "\n".join(state["tasks"])
    )
    reordered = clean_lines(model.invoke(prompt).content)

    # Safety check: accept the new order only if the model kept every task.
    if sorted(t.lower() for t in reordered) == sorted(t.lower() for t in state["tasks"]):
        print("Queue after prioritizing:", reordered)
        return {"tasks": reordered}
    print("Prioritizer changed the task set, keeping the original order.")
    return {}


def should_continue(state: BabyState) -> str:
    if state["tasks"] and state["iteration"] < MAX_ITERATIONS:
        return "execute"
    return END


graph = StateGraph(BabyState)
graph.add_node("execute", execute)
graph.add_node("create", create)
graph.add_node("prioritize", prioritize)
graph.set_entry_point("execute")
graph.add_edge("execute", "create")
graph.add_edge("create", "prioritize")
graph.add_conditional_edges("prioritize", should_continue)
app = graph.compile()


def main():
    objective = "Prepare a short revision plan for the CNN module of a deep learning course."
    final_state = app.invoke(
        {
            "objective": objective,
            "tasks": ["Write a list of the main topics needed for this objective"],
            "done": [],
            "iteration": 0,
        }
    )

    print("\n--- Completed tasks ---")
    for i, d in enumerate(final_state["done"], 1):
        print(f"{i}. {d['task']}\n   {d['result']}")
    if final_state["tasks"]:
        print("\nStill in the queue when the loop stopped:")
        for t in final_state["tasks"]:
            print(f"- {t}")


if __name__ == "__main__":
    main()