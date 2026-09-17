"""
Pattern 1: Simple Tool-Calling (no loop)

The model gets one chance to decide whether to call a tool. We catch
that decision ourselves and control exactly what happens next. There
is no automatic loop back to the model after the tool runs, unless
we write that step ourselves.

When to use: a single, predictable tool call inside a flow where you
want full control over what happens after the tool runs.

Run:
    python main.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv



from langchain.chat_models import init_chat_model
from shared.tools import calculator
load_dotenv()


def main():
    model = init_chat_model("google_genai:gemini-2.5-flash", temperature=0)
    model_with_tools = model.bind_tools([calculator])

    user_message = "What is 42 times 17, plus 8?"
    response = model_with_tools.invoke(user_message)

    if response.tool_calls:
        call = response.tool_calls[0]
        print(f"Model wants to call: {call['name']}({call['args']})")

        # We decide what happens next. No automatic loop back to the model.
        result = calculator.invoke(call["args"])
        print(f"Tool result: {result}")

        # Optional: feed the result back once, ourselves, for a natural-language answer.
        follow_up = model.invoke(
            f"The calculation result was {result}. Answer the user's question: {user_message}"
        )
        print(f"Final answer: {follow_up.content}")
    else:
        print(f"Model answered directly: {response.content}")


if __name__ == "__main__":
    main()
