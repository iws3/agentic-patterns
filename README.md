# Agentic Patterns

Five ways to structure how an LLM plans and executes actions with
tools, each with a runnable LangChain and LangGraph example. Built as
a learning reference: read the pattern, run the code, see the loop
happen.

Full write-up with diagrams and trade-offs: [`docs/AGENTIC_PATTERNS.md`](docs/AGENTIC_PATTERNS.md)

## Patterns covered

| # | Pattern | Folder | Use when |
|---|---|---|---|
| 1 | Simple Tool-Calling | [`01_simple_tool_calling/`](01_simple_tool_calling) | One predictable tool call, you control the flow after |
| 2 | ReAct | [`02_react_agent/`](02_react_agent) | Open-ended requests, unknown number or order of tool calls |
| 3 | Plan-and-Execute | [`03_plan_and_execute/`](03_plan_and_execute) | Known multi-step workflow, efficiency or a reviewable plan matters |
| 4 | Reflection | [`04_reflection/`](04_reflection) | Output quality matters more than speed |
| 5 | Multi-Agent (Supervisor) | [`05_multi_agent_supervisor/`](05_multi_agent_supervisor) | Toolset has grown too large for one prompt |

## Folder structure

```
agentic-patterns/
├── docs/
│   └── AGENTIC_PATTERNS.md      # full guide: TOC, diagrams, trade-offs, code
├── shared/
│   └── tools.py                 # tools reused across every example
├── 01_simple_tool_calling/
│   ├── README.md
│   └── main.py
├── 02_react_agent/
│   ├── README.md
│   └── main.py
├── 03_plan_and_execute/
│   ├── README.md
│   └── main.py
├── 04_reflection/
│   ├── README.md
│   └── main.py
├── 05_multi_agent_supervisor/
│   ├── README.md
│   └── main.py
├── requirements.txt
├── .env.example
└── LICENSE
```

## Setup

```bash
git clone https://github.com/<your-username>/agentic-patterns.git
cd agentic-patterns
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then fill in your API key
```

Each `main.py` uses `init_chat_model("gpt-4o-mini")` by default. Swap
the model string and provider package to use a different provider,
`init_chat_model` supports OpenAI, Anthropic, Google, and others
through the same interface.

## Running an example

```bash
cd 02_react_agent
python main.py
```

Every folder is runnable on its own and has a short README explaining
what to expect and why the pattern fits (or does not fit) a given
task.

## Why this exists

These five patterns come up constantly once you move past a single
tool call and start building assistants that need to decide things
for themselves. This repo exists so the difference between them is
something you can run and watch, not just read about.

## License

MIT, see [LICENSE](LICENSE).
