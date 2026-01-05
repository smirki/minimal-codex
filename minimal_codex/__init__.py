"""Minimal Codex Agent - A 1:1 replica of Codex CLI's autonomous agent logic."""

from .agent import CodexAgent, convert_to_atif, main
from .features import Features, Feature
from .tools import (
    TOOLS,
    CORE_TOOLS,
    WEB_SEARCH_TOOL,
    EXEC_COMMAND_TOOL,
    WRITE_STDIN_TOOL,
    SAVE_PLAN_TOOL,
    create_invoke_subagent_tool,
    execute_tool,
)
from .prompts import get_system_prompt
from .context import build_initial_messages
from .apply_patch import apply_patch
from .streaming import StreamController, DeltaCollector, stream_response
from .pty_shell import PtySessionManager, PtySession, HAS_PTY
from .subagents import SubagentConfig, SubagentSession, SubagentManager
from .plan_manager import PlanManager
from .prompt_templates import load_prompt_template, resolve_template

__version__ = "0.3.0"

__all__ = [
    # Agent
    "CodexAgent",
    "convert_to_atif",
    "main",
    # Features
    "Features",
    "Feature",
    # Tools
    "TOOLS",
    "CORE_TOOLS",
    "WEB_SEARCH_TOOL",
    "EXEC_COMMAND_TOOL",
    "WRITE_STDIN_TOOL",
    "SAVE_PLAN_TOOL",
    "create_invoke_subagent_tool",
    "execute_tool",
    # Context/Prompts
    "get_system_prompt",
    "build_initial_messages",
    "apply_patch",
    # Streaming
    "StreamController",
    "DeltaCollector",
    "stream_response",
    # PTY
    "PtySessionManager",
    "PtySession",
    "HAS_PTY",
    # Subagents
    "SubagentConfig",
    "SubagentSession",
    "SubagentManager",
    # Plan Mode
    "PlanManager",
    # Prompt Templates
    "load_prompt_template",
    "resolve_template",
]
