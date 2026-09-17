# 1. Simple Tool-Calling

The model gets one chance to decide whether to call a tool. Your code
catches that decision and controls everything that happens next.

**Use when:** a single, predictable tool call inside a flow where you
want full control over what happens after, for example extracting
structured data and then validating or saving it yourself.

**Trade-off:** no multi-step reasoning. If a task needs two tools in
sequence, you hardcode that sequence.

Run it:

```bash
pip install -r ../requirements.txt
python main.py
```
