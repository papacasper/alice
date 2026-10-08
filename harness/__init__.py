"""Modular local-LLM agent harness: llm (Ollama client) + tools (registry) + agent (loop)."""
from .agent import Agent, Result, DEFAULT_SYSTEM
from .llm import OllamaClient, LLMError
from .tools import Tool, Toolbox, tool

__all__ = ["Agent", "Result", "DEFAULT_SYSTEM", "OllamaClient", "LLMError", "Tool", "Toolbox", "tool"]
