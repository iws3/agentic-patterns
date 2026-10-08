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

"""


import os
import sys
from typing import List, TypedDict
from dotenv import load_dotenv


from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
load_dotenv()

model=init_chat_model("google_genai:gemini-2.5-flash", temperature=0)

class PlanState(TypedDict):
    task: str
    steps: List[str]
    current_step: int
    results: List[str]

def planner(state:PlanState)->dict:
    prompt=(
        "Break this task into 2 short, concrete steps, one per line"
    f"No numbering, no extra text:\n\n{state['task']}")
    response=model.invoke(prompt)
    steps=[line.strip() for line in response.content.strip().split("\n") if line.strip()]
    print("Plan: ")
    for i, s in enumerate(steps):
        print(f" {i+1}, {s}")

        return {"steps": steps, "current_step": 0, "results": []}
    
    
