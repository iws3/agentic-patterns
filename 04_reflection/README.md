# 4. Reflection (self-critique)

The model drafts an answer, critiques its own draft, and revises if
the critique found problems. Loops until it passes or hits a cap.

**Use when:** output quality matters more than speed, and first
drafts are known to be rough, long-form writing, prompts for other
tools, or code.

**Trade-off:** at least doubles cost and latency per task (draft plus
critique, often more with revisions). Needs its own retry cap.

Run it:

```bash
pip install -r ../requirements.txt
python main.py
```
