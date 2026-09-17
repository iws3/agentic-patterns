# 2. ReAct (Reason, Act, Observe, repeat)

The model loops: reason about what to do, take an action, observe the
result, decide if it is done. `create_react_agent` builds this loop
for you.

**Use when:** you cannot predict in advance which tools, if any, a
message will need. This is the general-purpose default.

**Trade-off:** the number of loop iterations varies per request, so
cost and latency are less predictable. Always set a `recursion_limit`.

Run it:

```bash
pip install -r ../requirements.txt
python main.py
```
