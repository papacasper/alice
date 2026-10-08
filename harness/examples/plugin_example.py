"""Example plugin: python3 -m harness --tools-file harness/examples/plugin_example.py "..." """
import datetime
from harness import tool

@tool
def today() -> str:
    """Return today's date (YYYY-MM-DD)."""
    return datetime.date.today().isoformat()

@tool
def word_count(text: str) -> str:
    """Count the words in a piece of text.

    Args:
        text: the text to count
    """
    return str(len(text.split()))
