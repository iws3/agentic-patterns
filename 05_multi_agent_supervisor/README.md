# 5. Multi-Agent (Supervisor / Orchestrator-Worker)

A supervisor model routes each request to a specialized sub-agent,
each with its own narrow toolset and prompt.

**Use when:** your toolset or task variety has grown too large for
one system prompt to describe well, or the model starts picking the
wrong tool from too many overlapping options.

**Trade-off:** more moving parts, extra latency from the routing
call itself, and routing mistakes are a new failure mode.

Run it:

```bash
pip install -r ../requirements.txt
python main.py
```
