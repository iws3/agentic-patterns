# 3. Plan-and-Execute

The model writes a full plan up front, then executes each step in
order, rather than re-deciding from scratch after every action.

**Use when:** the steps are largely known in advance and efficiency
matters, or you want a plan a user can review before anything runs.

**Trade-off:** less adaptive mid-task. If step 3 changes what step 5
should be, you need explicit re-planning logic to catch that.

Run it:

```bash
pip install -r ../requirements.txt
python main.py
```
