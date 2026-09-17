"""
Shared tools used across the agentic pattern examples.

These are intentionally simple and mostly dependency-free so every
example runs the same way regardless of which pattern it demonstrates.
Swap web_search for a real provider (e.g. Tavily) when you move past
learning the patterns and into a real project.
"""

from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city. Use this when the user asks about weather conditions."""
    fake_weather = {
        "lagos": "31C, humid, scattered showers expected this afternoon",
        "douala": "29C, partly cloudy",
        "nairobi": "22C, clear skies",
        "yaounde": "27C, light rain",
    }
    return fake_weather.get(city.lower(), f"No weather data found for {city}.")


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use this for facts that may have changed recently."""
    return (
        f"[Mock search result for '{query}']: This is placeholder data. "
        "Wire this up to a real search API (e.g. Tavily) for production use."
    )


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (4 + 3)'. Use this for any math the user asks for."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression))  # noqa: S307 - input is restricted to a safe character set above
    except Exception as e:
        return f"Error evaluating expression: {e}"
