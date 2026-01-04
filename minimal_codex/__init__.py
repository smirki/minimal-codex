"""Minimal Codex Agent - A 1:1 replica of Codex CLI's autonomous agent logic."""

from .agent import CodexAgent, convert_to_atif, main
from .tools import TOOLS
from .prompts import get_system_prompt
from .context import build_initial_messages
from .apply_patch import apply_patch

__version__ = "0.1.0"

__all__ = [
    "CodexAgent",
    "convert_to_atif",
    "main",
    "TOOLS",
    "get_system_prompt",
    "build_initial_messages",
    "apply_patch",
]
