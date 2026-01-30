"""Main agent class for the Minimal Codex Agent.

Implements the core agent loop matching Codex CLI's logic:
- API calls with retry and backoff
- Tool execution (parallel/sequential)
- Trajectory recording for ATIF format
- Conversation management
- Feature flags (PTY shell, streaming, web search)
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import litellm
from litellm import completion

from .apply_patch import apply_patch
from .context import build_initial_messages
from .features import Features, Feature
from .plan_manager import PlanManager
from .prompts import get_system_prompt
from .streaming import StreamController, stream_response
from .subagents import SubagentManager
from .pty_shell import PtySessionManager, HAS_PTY
from .tools import (
    CORE_TOOLS,
    PARALLEL_TOOLS,
    WEB_SEARCH_TOOL,
    EXEC_COMMAND_TOOL,
    WRITE_STDIN_TOOL,
    SAVE_PLAN_TOOL,
    create_invoke_subagent_tool,
    execute_shell,
    execute_web_search,
    read_file,
    list_dir,
    grep_files,
    update_plan,
    truncate_output,
    HAS_WEB_SEARCH,
    tool_supports_parallel,
)

import threading
from contextlib import contextmanager


class RwLock:
    """Read-Write Lock for tool execution (matches Codex's tokio::sync::RwLock behavior).

    - Multiple readers can hold the lock simultaneously
    - Writers get exclusive access
    - Writers wait for all readers to finish

    This enables safe parallel execution of read-only tools (read_file, list_dir, grep_files)
    while ensuring mutating tools (shell, apply_patch) get exclusive access.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._readers = 0
        self._writer_waiting = False
        self._read_ready = threading.Condition(self._lock)
        self._write_ready = threading.Condition(self._lock)

    @contextmanager
    def read_lock(self):
        """Acquire read lock (shared access for parallel tools)."""
        with self._lock:
            while self._writer_waiting:
                self._read_ready.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._lock:
                self._readers -= 1
                if self._readers == 0:
                    self._write_ready.notify()

    @contextmanager
    def write_lock(self):
        """Acquire write lock (exclusive access for mutating tools)."""
        with self._lock:
            self._writer_waiting = True
            while self._readers > 0:
                self._write_ready.wait()
        try:
            yield
        finally:
            with self._lock:
                self._writer_waiting = False
                self._read_ready.notify_all()


# LiteLLM configuration - important for compatibility with various APIs
litellm.drop_params = True

# ============================================================================
# Context Compaction Constants (from Codex's compact.rs and truncate.rs)
# ============================================================================

# Exact constant from Codex's truncate.rs
APPROX_BYTES_PER_TOKEN = 4
COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000
COMPACT_USER_MESSAGE_MAX_BYTES = COMPACT_USER_MESSAGE_MAX_TOKENS * APPROX_BYTES_PER_TOKEN  # 80,000

# Retry constants (exact Codex values from util.rs)
INITIAL_DELAY_MS = 200  # 200ms starting delay
BACKOFF_FACTOR = 2.0
DEFAULT_STREAM_MAX_RETRIES = 5
DEFAULT_REQUEST_MAX_RETRIES = 4
STREAM_IDLE_TIMEOUT_MS = 300_000  # 5 minutes

# Retryable HTTP status codes
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Exact prompt from compact/prompt.md
COMPACT_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."""

# Exact prefix from compact/summary_prefix.md
SUMMARY_PREFIX = """Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used by that language model. Use this to build on the work that has already been done and avoid duplicating work. Here is the summary produced by the other language model, use the information in this summary to assist with your own analysis:"""


# ============================================================================
# Context Compaction Functions (from Codex's compact.rs)
# ============================================================================

def approx_token_count(text: str) -> int:
    """Approximate token count (exact Codex formula from truncate.rs)."""
    byte_len = len(text.encode('utf-8'))
    return (byte_len + APPROX_BYTES_PER_TOKEN - 1) // APPROX_BYTES_PER_TOKEN


def backoff(attempt: int) -> float:
    """Calculate backoff delay with jitter (exact Codex algorithm from util.rs).

    Formula: base_delay * 2^(attempt-1) * jitter
    Where jitter is 0.9-1.1 (±10%)
    """
    exp = BACKOFF_FACTOR ** max(0, attempt - 1)
    base_ms = INITIAL_DELAY_MS * exp
    jitter = random.uniform(0.9, 1.1)
    return (base_ms * jitter) / 1000.0  # Convert to seconds


def _is_summary_message(msg: dict) -> bool:
    """Check if message is a previous compaction summary (like Codex's is_summary_message).

    Filters out previous summaries to prevent summary-of-summary bloat during
    repeated compactions.
    """
    content = msg.get("content", "")
    if isinstance(content, list):
        # Handle multi-part content
        for part in content:
            if part.get("type") == "text":
                text = part.get("text", "")
                if text.strip().startswith(SUMMARY_PREFIX.strip()[:50]):
                    return True
        return False
    return str(content).strip().startswith(SUMMARY_PREFIX.strip()[:50])


def collect_user_messages(
    messages: list[dict],
    max_bytes: int = COMPACT_USER_MESSAGE_MAX_BYTES
) -> str:
    """Extract user messages from history up to max_bytes.

    This matches Codex's collect_user_messages() exactly:
    - Filters out previous summary messages (prevents summary-of-summary bloat)
    - Processes in REVERSE order (newest first) to prioritize recent context
    - Stops when byte limit is reached
    - Reverses back to chronological order for the summary
    """
    collected = []
    total_bytes = 0

    # Filter out previous summary messages (Codex's is_summary_message check)
    user_messages = [
        msg for msg in messages
        if msg.get("role") == "user"
        and not _is_summary_message(msg)
    ]

    # Process in REVERSE order (newest first) like real Codex
    for msg in reversed(user_messages):
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle multi-part messages (text + images)
            content = " ".join(
                part.get("text", "") for part in content
                if part.get("type") == "text"
            )

        content_bytes = len(content.encode('utf-8'))

        if total_bytes + content_bytes > max_bytes:
            # Truncate this message to fit remaining budget
            remaining = max_bytes - total_bytes
            if remaining > 100:  # Only include if meaningful
                truncated = content.encode('utf-8')[:remaining].decode('utf-8', errors='ignore')
                collected.append(truncated)
            break

        collected.append(content)
        total_bytes += content_bytes

    # Reverse back to chronological order for summary
    collected.reverse()
    return "\n\n".join(collected)


def build_compacted_history(
    original_messages: list[dict],
    summary: str
) -> list[dict]:
    """Build new conversation history with summary.

    Matches Codex's build_compacted_history():
    - Keep system message(s) at the start
    - Add summary as a user message with prefix
    - Include preserved user messages
    """
    compacted = []

    # 1. Keep system messages at start
    for msg in original_messages:
        if msg.get("role") == "system":
            compacted.append(msg)
        else:
            break  # Stop at first non-system message

    # 2. Add summary with prefix
    summary_message = {
        "role": "user",
        "content": f"{SUMMARY_PREFIX}\n\n{summary}"
    }
    compacted.append(summary_message)

    # 3. Collect user messages up to limit
    user_content = collect_user_messages(original_messages, COMPACT_USER_MESSAGE_MAX_BYTES)
    if user_content:
        compacted.append({
            "role": "user",
            "content": f"[Preserved user context]\n{user_content}"
        })

    return compacted


class CodexAgent:
    """Minimal Codex Agent - 1:1 replica of Codex CLI's autonomous logic."""

    def __init__(
        self,
        model: str,
        cwd: str = ".",
        context_window: int = 128000,
        features: Optional[Features] = None,
    ):
        """Initialize the agent.

        Args:
            model: Model name to use (passed to LiteLLM)
            cwd: Working directory for the agent
            context_window: Model's context window size (for compaction threshold)
            features: Feature flags (all enabled by default)
        """
        self.model = model
        self.cwd = Path(cwd).resolve()
        self.messages = []
        self.trajectory = []  # Internal format for ATIF conversion
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_cost = 0.0
        self.plan = []  # Current plan state

        # Incremental save paths (set by run() from CLI args)
        self.trajectory_output_path = None  # Path to trajectory.json in mounted volume
        self.output_json_path = None        # Path to output.json in mounted volume
        self.session_id = str(uuid.uuid4())  # Stable session ID for ATIF

        # Feature flags (all enabled by default)
        self.features = features or Features()

        # PTY session manager (if PTY feature enabled and available)
        self.pty_manager: Optional[PtySessionManager] = None
        if self.features.enabled(Feature.PTY_SHELL) and HAS_PTY:
            self.pty_manager = PtySessionManager()

        # Context compaction thresholds (from Codex config)
        self.context_window = context_window
        self.compaction_threshold = 0.8  # Compact at 80% of context
        self.last_compaction_tokens = 0

        # Subagent manager (if SUBAGENTS feature enabled)
        self.subagent_manager: Optional[SubagentManager] = None
        if self.features.enabled(Feature.SUBAGENTS):
            self.subagent_manager = SubagentManager(
                cwd=self.cwd,
                model=self.model,
                all_tools=self._get_base_tools(),
            )

        # Plan manager (if PLAN_MODE feature enabled)
        self.plan_manager: Optional[PlanManager] = None
        self.current_plan_path: Optional[Path] = None
        self.current_task: str = ""  # Track current task for save_plan
        self.trajectory_output_dir: Optional[Path] = None  # Set by run() for subagent trajectories
        if self.features.enabled(Feature.PLAN_MODE):
            self.plan_manager = PlanManager(self.cwd)

        # Completion detection (matches real Codex's needs_follow_up flag)
        self.needs_follow_up = False

        # Smart read/write lock for tool execution (matches Codex's tokio::sync::RwLock)
        self._tool_lock = RwLock()

    def _get_base_tools(self) -> list:
        """Get base tools without subagent invocation.

        Used to initialize SubagentManager (subagents get these tools
        but NOT invoke_subagent to prevent nested delegation).
        """
        tools = CORE_TOOLS.copy()

        # Add web search if enabled and available
        if self.features.enabled(Feature.WEB_SEARCH) and HAS_WEB_SEARCH:
            tools.append(WEB_SEARCH_TOOL)

        # Add PTY tools if enabled and available
        if self.features.enabled(Feature.PTY_SHELL) and HAS_PTY:
            tools.append(EXEC_COMMAND_TOOL)
            tools.append(WRITE_STDIN_TOOL)

        return tools

    def _get_tools(self) -> list:
        """Get the full list of tools including invoke_subagent."""
        tools = self._get_base_tools()

        # Add invoke_subagent if subagents enabled
        if self.subagent_manager:
            invoke_tool = create_invoke_subagent_tool(
                self.subagent_manager.get_available_subagents()
            )
            tools.append(invoke_tool)

        return tools

    def run(self, task: str, max_turns: Optional[int] = None, use_plan_mode: bool = False,
            trajectory_path: Optional[str] = None, output_path: Optional[str] = None) -> dict:
        """Run the agent until task completion.

        Args:
            task: The task to perform
            max_turns: Maximum turns (None = unlimited, uses compaction like real Codex)
            use_plan_mode: If True and PLAN_MODE enabled, use autonomous planning
            trajectory_path: Path where main trajectory will be saved (for subagent trajectories)
            output_path: Path where output.json will be saved

        Returns:
            Dict with output, trajectory, token counts, etc.
        """
        # Track current task for save_plan
        self.current_task = task
        self.trajectory_output_dir = Path(trajectory_path).parent if trajectory_path else None

        # Set paths for incremental saving (survives timeout/crash)
        self.trajectory_output_path = trajectory_path
        self.output_json_path = output_path

        # Check if plan mode should be used
        if use_plan_mode and self.features.enabled(Feature.PLAN_MODE) and self.subagent_manager:
            return self._run_with_plan_mode(task, max_turns)

        # Standard execution
        return self._run_standard(task, max_turns)

    def _run_standard(self, task: str, max_turns: Optional[int] = None) -> dict:
        """Run standard agent loop without plan mode.

        Matches real Codex behavior:
        - Unlimited turns by default (uses compaction instead of stopping)
        - needs_follow_up flag for completion detection
        - Optional max_turns safety limit

        Args:
            task: The task to perform
            max_turns: Maximum turns (None = unlimited like real Codex)

        Returns:
            Dict with output, trajectory, token counts, etc.
        """
        # Initialize conversation with context
        self.messages = build_initial_messages(self.cwd, task)

        # Record initial messages to trajectory
        for msg in self.messages:
            self._record_message(msg["role"], msg["content"])

        turn = 0
        while True:
            # Optional safety limit (can be set via parameter)
            if max_turns is not None and turn >= max_turns:
                print(f"Warning: Reached max_turns limit ({max_turns})")
                return {
                    "error": "Max turns exceeded",
                    "trajectory": self.trajectory,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                }

            # Check for compaction before API call (like real Codex - compact instead of stopping)
            if self.should_compact():
                print(f"[Token limit approaching at turn {turn}, compacting context...]")
                self.compact_conversation()

            try:
                response = self._call_api_with_retry()
            except Exception as e:
                return {
                    "error": str(e),
                    "trajectory": self.trajectory,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                }

            assistant_message = response.choices[0].message

            # Track token usage
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "cached_tokens": getattr(response.usage, 'cached_tokens', 0) if hasattr(response.usage, 'cached_tokens') else 0,
                }
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)
                self.total_cached_tokens += usage.get("cached_tokens", 0)

            # Add to conversation history (use explicit serialization per Codex pattern)
            self.messages.append(self._serialize_assistant_message(assistant_message))

            # Record to trajectory
            self._record_assistant(assistant_message, usage)

            # Completion detection using needs_follow_up (matches real Codex)
            if assistant_message.tool_calls:
                self.needs_follow_up = True  # Tool calls requested = need follow-up

                # Execute tool calls
                tool_results = self._execute_tool_calls(assistant_message.tool_calls)

                # Add results to conversation and trajectory
                for result in tool_results:
                    self.messages.append(result)
                    self._record_tool_result(result["tool_call_id"], result["content"])
            else:
                # No tool calls in response = model is done
                self.needs_follow_up = False

            # Exit condition (matches real Codex)
            if not self.needs_follow_up:
                return {
                    "output": assistant_message.content,
                    "trajectory": self.trajectory,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "cached_tokens": self.total_cached_tokens,
                    "usage": {
                        "input_tokens": self.total_input_tokens,
                        "cached_input_tokens": self.total_cached_tokens,
                        "output_tokens": self.total_output_tokens,
                    }
                }

            turn += 1

    def _run_with_plan_mode(self, task: str, max_turns: Optional[int]) -> dict:
        """Plan mode - just adds read-only planning prompt to system message.

        Following Claude Code's pattern: plan mode is just a prompt, not phases.
        The main agent decides when/if to use subagents.

        Args:
            task: The task to perform
            max_turns: Maximum turns (None = unlimited)

        Returns:
            Dict with output, trajectory, token counts, etc.
        """
        from .prompt_templates import load_prompt_template

        print("[Plan Mode] Adding planning guidance to system prompt")

        # Initialize conversation with context
        self.messages = build_initial_messages(self.cwd, task)

        # Load planning guidance prompt and add to system message
        planning_guidance = load_prompt_template("plan_mode_main")
        self.messages[0]["content"] += "\n\n" + planning_guidance

        # Record initial messages to trajectory
        for msg in self.messages:
            self._record_message(msg["role"], msg["content"])

        # Run standard loop - main agent makes all decisions
        return self._run_standard_loop(max_turns)

    def _run_standard_loop(self, max_turns: Optional[int]) -> dict:
        """Run the standard agent loop (shared by both modes).

        Matches real Codex behavior:
        - Unlimited turns by default (uses compaction instead of stopping)
        - needs_follow_up flag for completion detection

        Args:
            max_turns: Maximum turns (None = unlimited like real Codex)

        Returns:
            Dict with output, trajectory, token counts, etc.
        """
        turn = 0
        while True:
            # Optional safety limit
            if max_turns is not None and turn >= max_turns:
                print(f"Warning: Reached max_turns limit ({max_turns})")
                return {
                    "error": "Max turns exceeded",
                    "trajectory": self.trajectory,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                }

            # Check for compaction before API call (like real Codex)
            if self.should_compact():
                print(f"[Token limit approaching at turn {turn}, compacting context...]")
                self.compact_conversation()

            try:
                response = self._call_api_with_retry()
            except Exception as e:
                return {
                    "error": str(e),
                    "trajectory": self.trajectory,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                }

            assistant_message = response.choices[0].message

            # Track token usage
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "cached_tokens": getattr(response.usage, 'cached_tokens', 0) if hasattr(response.usage, 'cached_tokens') else 0,
                }
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)
                self.total_cached_tokens += usage.get("cached_tokens", 0)

            # Add to conversation history
            self.messages.append(self._serialize_assistant_message(assistant_message))

            # Record to trajectory
            self._record_assistant(assistant_message, usage)

            # Completion detection using needs_follow_up (matches real Codex)
            if assistant_message.tool_calls:
                self.needs_follow_up = True  # Tool calls requested = need follow-up

                # Execute tool calls
                tool_results = self._execute_tool_calls(assistant_message.tool_calls)

                # Add results to conversation and trajectory
                for result in tool_results:
                    self.messages.append(result)
                    self._record_tool_result(result["tool_call_id"], result["content"])
            else:
                # No tool calls in response = model is done
                self.needs_follow_up = False

            # Exit condition (matches real Codex)
            if not self.needs_follow_up:
                return {
                    "output": assistant_message.content,
                    "trajectory": self.trajectory,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "cached_tokens": self.total_cached_tokens,
                    "usage": {
                        "input_tokens": self.total_input_tokens,
                        "cached_input_tokens": self.total_cached_tokens,
                        "output_tokens": self.total_output_tokens,
                    }
                }

            turn += 1

    def _save_subagent_trajectories(self, sessions: list) -> list[str]:
        """Save subagent sessions to trajectory files in ATIF format.

        Saves next to the main trajectory.json file with names like:
        - trajectory_explore_1.json
        - trajectory_plan_1.json

        Args:
            sessions: List of (type, index, agent_id, session) tuples

        Returns:
            List of trajectory file paths
        """
        trajectory_files = []

        # Use the same directory as main trajectory, or fall back to .tessa/trajectories
        if self.trajectory_output_dir:
            trajectories_dir = self.trajectory_output_dir
        else:
            trajectories_dir = self.cwd / ".tessa" / "trajectories"

        trajectories_dir.mkdir(parents=True, exist_ok=True)

        for subagent_type, index, agent_id, session in sessions:
            # Convert session messages to ATIF format
            atif_trajectory = convert_to_atif(
                trajectory=session.messages,
                model_name=self.model,
                agent_name=f"minimal-codex-{subagent_type}",
                agent_version="0.3.0",
            )

            # Add subagent metadata
            atif_trajectory["subagent"] = {
                "type": subagent_type,
                "index": index,
                "agent_id": agent_id,
            }

            filename = f"trajectory_{subagent_type}_{index}.json"
            filepath = trajectories_dir / filename
            filepath.write_text(json.dumps(atif_trajectory, indent=2, default=str), encoding="utf-8")
            trajectory_files.append(str(filepath))

        return trajectory_files

    def _parse_synthesis_result(self, result: str) -> tuple[list, list]:
        """Parse the synthesis LLM output into steps and critical files.

        Args:
            result: Raw LLM output with STEPS: and CRITICAL_FILES: sections

        Returns:
            (steps, critical_files) - List of step dicts and list of file paths
        """
        steps = []
        critical_files = []

        lines = result.split("\n")
        current_section = None

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.upper().startswith("STEPS:"):
                current_section = "steps"
                continue
            elif line_stripped.upper().startswith("CRITICAL_FILES:"):
                current_section = "files"
                continue
            elif line_stripped.upper().startswith("NOTES:"):
                current_section = "notes"
                continue

            if current_section == "steps" and line_stripped:
                # Parse numbered step like "1. Do something"
                if line_stripped[0].isdigit() and "." in line_stripped:
                    step_text = line_stripped.split(".", 1)[-1].strip()
                    if step_text:
                        steps.append({"step": step_text, "status": "pending"})
                elif line_stripped.startswith("-"):
                    step_text = line_stripped[1:].strip()
                    if step_text:
                        steps.append({"step": step_text, "status": "pending"})

            elif current_section == "files" and line_stripped:
                # Parse file path like "- file.py" or just "file.py"
                if line_stripped.startswith("-"):
                    file_path = line_stripped[1:].strip().strip("`")
                else:
                    file_path = line_stripped.strip("`")
                if file_path and not file_path.upper().startswith("NOTES"):
                    critical_files.append(file_path)

        return steps, critical_files


    def _execute_with_plan(self, task: str, max_turns: int) -> dict:
        """Execute task with plan loaded in context.

        Args:
            task: The original task
            max_turns: Maximum turns for execution

        Returns:
            Dict with output, trajectory, token counts, etc.
        """
        plan_context = self.plan_manager.get_plan_context(self.current_plan_path)

        # Rebuild context with plan injected
        self.messages = build_initial_messages(self.cwd, f"{task}\n\n{plan_context}")

        # Record plan injection to trajectory
        self._record_message("system", f"[Plan Mode] Executing with plan:\n{plan_context}")

        # Run standard execution loop
        for turn in range(max_turns):
            # Check for compaction - re-inject plan if compacted
            if self.compact_conversation():
                print(f"[Compacted context at turn {turn}]")
                self._inject_plan_on_compaction()

            try:
                response = self._call_api_with_retry()
            except Exception as e:
                return {
                    "error": str(e),
                    "trajectory": self.trajectory,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                }

            assistant_message = response.choices[0].message

            # Track token usage
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "cached_tokens": getattr(response.usage, 'cached_tokens', 0) if hasattr(response.usage, 'cached_tokens') else 0,
                }
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)
                self.total_cached_tokens += usage.get("cached_tokens", 0)

            # Add to conversation history (use explicit serialization per Codex pattern)
            self.messages.append(self._serialize_assistant_message(assistant_message))

            # Record to trajectory
            self._record_assistant(assistant_message, usage)

            # Check if done (no tool calls)
            if not assistant_message.tool_calls:
                return {
                    "output": assistant_message.content,
                    "trajectory": self.trajectory,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "cached_tokens": self.total_cached_tokens,
                    "usage": {
                        "input_tokens": self.total_input_tokens,
                        "cached_input_tokens": self.total_cached_tokens,
                        "output_tokens": self.total_output_tokens,
                    }
                }

            # Execute tool calls
            tool_results = self._execute_tool_calls(assistant_message.tool_calls)

            # Add results to conversation and trajectory
            for result in tool_results:
                self.messages.append(result)
                self._record_tool_result(result["tool_call_id"], result["content"])

        return {
            "error": "Max turns exceeded",
            "trajectory": self.trajectory,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }

    def _inject_plan_on_compaction(self):
        """Re-inject plan after context compaction."""
        if self.current_plan_path and self.plan_manager:
            plan_context = self.plan_manager.get_plan_context(self.current_plan_path)
            # Prepend plan to compacted context
            self.messages.insert(1, {
                "role": "user",
                "content": f"[Plan continues from file]\n{plan_context}",
            })

    def _call_api_with_retry(self, max_retries: int = DEFAULT_STREAM_MAX_RETRIES) -> Any:
        """Call API with exponential backoff retry (matches real Codex).

        Uses Codex's exact retry algorithm from util.rs:
        - Base delay: 200ms
        - Backoff factor: 2x
        - Jitter: ±10%
        - Default max retries: 5

        Supports streaming when Feature.STREAMING is enabled.

        Args:
            max_retries: Maximum number of retry attempts (default 5 like Codex)

        Returns:
            API response object
        """
        tools = self._get_tools()
        use_streaming = self.features.enabled(Feature.STREAMING)
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                if use_streaming:
                    # Streaming mode: collect chunks and output to stdout
                    response = completion(
                        model=self.model,
                        messages=self.messages,
                        tools=tools,
                        tool_choice="auto",
                        parallel_tool_calls=True,
                        stream=True,
                    )
                    return self._handle_streaming_response(response)
                else:
                    # Non-streaming mode
                    response = completion(
                        model=self.model,
                        messages=self.messages,
                        tools=tools,
                        tool_choice="auto",
                        parallel_tool_calls=True,
                    )
                    return response
            except Exception as e:
                last_exception = e

                # Check if retryable (like Codex's error classification)
                if not self._is_retryable_error(e):
                    raise  # Non-retryable errors fail immediately

                if attempt < max_retries:
                    # Use Codex's exact backoff algorithm
                    delay = backoff(attempt + 1)

                    # Log retry attempt (like Codex's warn! macro)
                    print(f"API error - retrying ({attempt + 1}/{max_retries} in {delay:.2f}s): {e}")

                    time.sleep(delay)
                else:
                    raise last_exception

        raise last_exception or Exception("Max retries exceeded")

    def _handle_streaming_response(self, response_stream) -> Any:
        """Handle streaming response, outputting tokens as they arrive.

        Returns a synthetic response object matching non-streaming format.
        """
        controller = StreamController()
        full_content = ""
        tool_calls_data = {}  # Accumulate tool call chunks
        usage_data = {}

        for chunk in response_stream:
            if not hasattr(chunk, 'choices') or not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Handle content streaming
            if hasattr(delta, 'content') and delta.content:
                full_content += delta.content
                controller.push(delta.content)

            # Handle tool calls streaming
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {
                            "id": "",
                            "function": {"name": "", "arguments": ""}
                        }
                    if tc.id:
                        tool_calls_data[idx]["id"] = tc.id
                    if hasattr(tc, 'function') and tc.function:
                        if tc.function.name:
                            tool_calls_data[idx]["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_data[idx]["function"]["arguments"] += tc.function.arguments

            # Capture usage from final chunk
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_data = {
                    "prompt_tokens": getattr(chunk.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(chunk.usage, 'completion_tokens', 0),
                }

        # Finalize streaming output
        controller.finalize()

        # Build synthetic response matching non-streaming format
        from types import SimpleNamespace

        tool_calls = None
        if tool_calls_data:
            tool_calls = []
            for idx in sorted(tool_calls_data.keys()):
                tc_data = tool_calls_data[idx]
                tool_calls.append(SimpleNamespace(
                    id=tc_data["id"],
                    function=SimpleNamespace(
                        name=tc_data["function"]["name"],
                        arguments=tc_data["function"]["arguments"]
                    )
                ))

        message = SimpleNamespace(
            content=full_content if full_content else None,
            tool_calls=tool_calls if tool_calls else None,
            # Note: model_dump is kept for compatibility but _serialize_assistant_message is used
            model_dump=lambda: {
                "role": "assistant",
                "content": None if tool_calls else (full_content if full_content else None),
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",  # REQUIRED per Codex pattern
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in (tool_calls or [])
                ] if tool_calls else None
            }
        )

        usage = SimpleNamespace(**usage_data) if usage_data else None

        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=usage
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable (matches Codex's error classification).

        Retryable: 429, 500, 502, 503, 504, timeout, connection errors
        Non-retryable: Authentication, invalid request, quota exceeded
        """
        error_str = str(error).lower()

        # Check for retryable HTTP status codes
        for code in RETRYABLE_STATUS_CODES:
            if str(code) in error_str:
                return True

        # Check for retryable error types
        retryable_keywords = [
            "timeout", "timed out",
            "connection", "connect",
            "temporary", "transient",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "rate limit",  # 429
            "stream",  # SSE stream errors
        ]

        return any(keyword in error_str for keyword in retryable_keywords)

    def _serialize_assistant_message(self, message) -> dict:
        """Serialize assistant message following Codex's exact Chat Completions format.

        This matches codex-rs/codex-api/src/requests/chat.rs exactly:
        - "type": "function" MUST be explicitly set in tool_calls
        - "content": null MUST be explicit (not omitted) when there are tool calls
        - Arguments kept as strings (already JSON-encoded)

        Args:
            message: LiteLLM message object with content and optional tool_calls

        Returns:
            Dict suitable for Chat Completions API
        """
        if not message.tool_calls:
            # Simple text message
            return {
                "role": "assistant",
                "content": message.content,
            }

        # Message with tool calls - Codex pattern from chat.rs lines 201-224
        return {
            "role": "assistant",
            "content": None,  # Codex explicitly sets null, not missing
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",  # REQUIRED - Codex always sets this explicitly
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]
        }

    # ========================================================================
    # Context Compaction Methods (from Codex's compact.rs)
    # ========================================================================

    def should_compact(self) -> bool:
        """Check if compaction is needed."""
        current_tokens = self._estimate_context_tokens()
        threshold_tokens = int(self.context_window * self.compaction_threshold)

        # Only compact if significantly over threshold
        # and we haven't just compacted
        return (current_tokens > threshold_tokens and
                current_tokens - self.last_compaction_tokens > 10000)

    def _estimate_context_tokens(self) -> int:
        """Estimate total tokens in conversation."""
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += approx_token_count(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        total += approx_token_count(part.get("text", ""))
        return total

    def compact_conversation(self) -> bool:
        """Perform context compaction using LLM summarization (matches real Codex).

        Includes overflow retry logic: if context still exceeds limits during
        compaction, trim oldest history items and retry (like Codex's compact.rs).

        Returns True if compaction was performed.
        """
        if not self.should_compact():
            return False

        # Collect user messages for summarization (newest-first with summary filtering)
        user_content = collect_user_messages(self.messages)
        if not user_content.strip():
            return False

        summary_messages = [
            {"role": "system", "content": COMPACT_PROMPT},
            {"role": "user", "content": user_content}
        ]

        max_retries = 3
        history_to_compact = list(self.messages)  # Copy for trimming

        for attempt in range(max_retries):
            try:
                # Call model for summarization
                response = completion(
                    model=self.model,
                    messages=summary_messages,
                    max_tokens=4000,  # Allow substantial summary
                    temperature=0.3,  # Lower temperature for consistent summarization
                )
                summary = response.choices[0].message.content

                # Track tokens used for summary
                if hasattr(response, 'usage') and response.usage:
                    self.total_input_tokens += getattr(response.usage, 'prompt_tokens', 0)
                    self.total_output_tokens += getattr(response.usage, 'completion_tokens', 0)

                if summary:
                    full_summary = f"{SUMMARY_PREFIX}\n\n{summary}"
                    # Build new compacted history
                    self.messages = build_compacted_history(self.messages, full_summary)
                    self.last_compaction_tokens = self._estimate_context_tokens()
                    print(f"Context compacted. New token estimate: {self.last_compaction_tokens}")
                    return True

                return False

            except Exception as e:
                error_str = str(e).lower()
                # Check for context window exceeded (like Codex's ContextWindowExceeded handling)
                if "context" in error_str or "token" in error_str or "length" in error_str:
                    if len(history_to_compact) > 1:
                        # Remove oldest item and retry (like Codex)
                        print(f"Context exceeded during compaction, trimming oldest item (attempt {attempt + 1})")
                        history_to_compact.pop(0)
                        user_content = collect_user_messages(history_to_compact)
                        summary_messages[1]["content"] = user_content
                        continue
                # Log error but don't fail - just continue with full history
                print(f"Warning: Compaction failed: {e}")
                return False

        return False

    def _format_history_for_summary(self) -> str:
        """Format conversation history for summarization."""
        parts = []
        for i, msg in enumerate(self.messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "[non-text content]")
                    for part in content
                )

            # Truncate very long messages for summary
            if len(content) > 5000:
                content = content[:2000] + f"\n...[{len(content)-4000} chars truncated]...\n" + content[-2000:]

            parts.append(f"[{role.upper()} {i+1}]\n{content}")

        return "\n\n---\n\n".join(parts)

    def _execute_tool_calls(self, tool_calls: list) -> list[dict]:
        """Execute tool calls with smart read/write locking (matches real Codex).

        Uses RwLock like Codex's tokio::sync::RwLock:
        - Read-only tools (read_file, list_dir, grep_files): acquire read lock, can run concurrently
        - Mutating tools (shell, apply_patch): acquire write lock, exclusive access

        Args:
            tool_calls: List of tool call objects from the API

        Returns:
            List of tool result dicts
        """
        results = []

        # Separate tools by parallelization capability (matches Codex's tool_supports_parallel)
        parallel_calls = []
        sequential_calls = []

        for tc in tool_calls:
            tool_name = tc.function.name
            if tool_supports_parallel(tool_name):
                parallel_calls.append(tc)
            else:
                sequential_calls.append(tc)

        # Execute parallel tools concurrently with read locks
        if parallel_calls:
            with ThreadPoolExecutor(max_workers=min(len(parallel_calls), 4)) as executor:
                def execute_with_read_lock(call):
                    with self._tool_lock.read_lock():
                        return self._execute_single_tool(call)

                futures = [executor.submit(execute_with_read_lock, tc) for tc in parallel_calls]
                for f in futures:
                    results.append(f.result())

        # Execute sequential tools one at a time with write locks (exclusive access)
        for tc in sequential_calls:
            with self._tool_lock.write_lock():
                result = self._execute_single_tool(tc)
                results.append(result)

        return results

    def _execute_single_tool(self, tool_call) -> dict:
        """Execute a single tool call and return result.

        Args:
            tool_call: Tool call object from API

        Returns:
            Dict with role, tool_call_id, content
        """
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            output = f"Error: Invalid JSON arguments: {e}"
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": truncate_output(output)
            }

        try:
            if name == "shell_command":
                output = execute_shell(args, self.cwd)
            elif name == "apply_patch":
                output = apply_patch(args.get("patch", ""), self.cwd)
            elif name == "update_plan":
                output, self.plan = update_plan(args, self.plan)
            elif name == "read_file":
                output = read_file(args, self.cwd)
            elif name == "list_dir":
                output = list_dir(args, self.cwd)
            elif name == "grep_files":
                output = grep_files(args, self.cwd)
            elif name == "web_search":
                output = execute_web_search(args)
            elif name == "exec_command":
                output = self._execute_pty_command(args)
            elif name == "write_stdin":
                output = self._execute_write_stdin(args)
            elif name == "invoke_subagent":
                output = self._execute_invoke_subagent(args)
            elif name == "save_plan":
                output = self._execute_save_plan(args)
            else:
                output = f"Unknown tool: {name}"
        except Exception as e:
            output = f"Error: {str(e)}"

        # Truncate if too large
        output = truncate_output(output)

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": output
        }

    def _execute_pty_command(self, args: dict) -> str:
        """Execute a command in a PTY session.

        Matches Codex's exec_command tool.
        """
        if not self.pty_manager:
            return "Error: PTY shell not available. Install pexpect (Unix) or pywinpty (Windows)."

        command = args.get("command", [])
        if isinstance(command, str):
            command = [command]

        workdir = args.get("workdir")
        cwd = Path(workdir).resolve() if workdir else self.cwd
        yield_time_ms = args.get("yield_time_ms", 2500)

        output, _, process_id, exit_code = self.pty_manager.exec_command(
            command=command,
            cwd=cwd,
            yield_time_ms=yield_time_ms,
        )

        # Format output with process info
        result_parts = [output] if output else []

        if process_id:
            result_parts.append(f"\n[process_id: {process_id}]")
            result_parts.append("[Process still running - use write_stdin to interact]")
        elif exit_code is not None:
            result_parts.append(f"\n[exit_code: {exit_code}]")

        return "\n".join(result_parts) if result_parts else "(no output)"

    def _execute_write_stdin(self, args: dict) -> str:
        """Write input to a running PTY session.

        Matches Codex's write_stdin tool.
        """
        if not self.pty_manager:
            return "Error: PTY shell not available."

        process_id = args.get("process_id", "")
        input_data = args.get("input", "")
        yield_time_ms = args.get("yield_time_ms", 2500)

        output, alive_process_id, exit_code = self.pty_manager.write_stdin(
            process_id=process_id,
            input_data=input_data,
            yield_time_ms=yield_time_ms,
        )

        # Format output with process info
        result_parts = [output] if output else []

        if alive_process_id:
            result_parts.append(f"\n[process_id: {alive_process_id}]")
            result_parts.append("[Process still running]")
        elif exit_code is not None:
            result_parts.append(f"\n[exit_code: {exit_code}]")
            result_parts.append("[Process exited]")

        return "\n".join(result_parts) if result_parts else "(no output)"

    def _execute_invoke_subagent(self, args: dict) -> str:
        """Invoke a subagent to handle a specialized task.

        Args:
            args: Tool arguments with name, task, optional max_turns, resume_id, context

        Returns:
            Subagent result with agent_id for potential resumption
        """
        if not self.subagent_manager:
            return "Error: Subagents feature is disabled"

        name = args.get("name", "")
        task = args.get("task", "")
        max_turns = args.get("max_turns", 10)
        resume_id = args.get("resume_id")
        context = args.get("context")  # Optional context from main agent

        result, agent_id, session = self.subagent_manager.invoke(
            name=name,
            task=task,
            max_turns=max_turns,
            resume_id=resume_id,
            context=context,
        )

        # Save trajectory for debugging (session is already saved by invoke)
        if session:
            traj_path = self.subagent_manager.transcripts_dir / f"agent-{agent_id}.jsonl"
            print(f"[Subagent] Trajectory saved: {traj_path}")

            # Mirror subagent transcript to output directory (survives timeout/crash)
            if self.trajectory_output_path and traj_path.exists():
                try:
                    output_subagents_dir = Path(self.trajectory_output_path).parent / "subagents"
                    output_subagents_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(traj_path, output_subagents_dir / traj_path.name)
                except Exception as e:
                    print(f"[Warning] Failed to mirror subagent transcript: {e}")

        # Include agent_id for potential resumption
        return f"{result}\n\n[agent_id: {agent_id}]"

    def _execute_save_plan(self, args: dict) -> str:
        """Save the implementation plan.

        Args:
            args: Tool arguments with steps and critical_files

        Returns:
            Confirmation message with plan path
        """
        if not self.plan_manager:
            return "Error: Plan mode is disabled"

        steps = args.get("steps", [])
        critical_files = args.get("critical_files", [])

        path = self.plan_manager.create_plan(
            task=self.current_task,
            steps=steps,
            critical_files=critical_files,
        )

        # Mirror plan to output directory (survives timeout/crash)
        if self.trajectory_output_path and path.exists():
            try:
                output_plans_dir = Path(self.trajectory_output_path).parent / "plans"
                output_plans_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, output_plans_dir / path.name)
            except Exception as e:
                print(f"[Warning] Failed to mirror plan: {e}")

        return f"Plan saved to: {path}"

    def _record_message(self, role: str, content: str, **kwargs):
        """Record a message with timestamp for trajectory."""
        self.trajectory.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        })
        self._flush_trajectory_to_disk()

    def _record_assistant(self, message, usage: dict = None):
        """Record assistant message with tool calls and usage."""
        entry = {
            "role": "assistant",
            "content": message.content or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]
        if usage:
            entry["usage"] = usage
        self.trajectory.append(entry)
        self._flush_trajectory_to_disk()

    def _record_tool_result(self, tool_call_id: str, content: str):
        """Record tool execution result."""
        self.trajectory.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._flush_trajectory_to_disk()

    # ========================================================================
    # Incremental Trajectory Saving (survives container timeout/crash)
    # ========================================================================

    def _flush_trajectory_to_disk(self):
        """Incrementally save trajectory to mounted output directory.

        This writes the current trajectory state to disk after each step,
        ensuring data survives container timeout or crash. The mounted
        /logs/agent/ directory syncs to the host in real-time.
        """
        if not self.trajectory_output_path:
            return

        try:
            atif = self._convert_to_atif_internal()
            path = Path(self.trajectory_output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(atif, indent=2))

            # Log first flush to confirm incremental saving is working
            if not hasattr(self, '_first_flush_logged'):
                self._first_flush_logged = True
                print(f"[Trajectory] Incremental saving enabled: {path}")

            # Also flush output.json with current usage stats
            self._flush_output_to_disk()
        except Exception as e:
            # Don't crash the agent if flush fails - just warn
            print(f"[Warning] Failed to flush trajectory: {e}")

    def _flush_output_to_disk(self):
        """Incrementally save output.json with current usage stats."""
        if not self.output_json_path:
            return

        try:
            output = {
                "output": None,  # Not complete yet
                "error": None,
                "status": "running",
                "usage": {
                    "input_tokens": self.total_input_tokens,
                    "cached_input_tokens": self.total_cached_tokens,
                    "output_tokens": self.total_output_tokens,
                }
            }
            path = Path(self.output_json_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(output, indent=2))
        except Exception:
            pass  # Silent fail for output - trajectory is more important

    def _convert_to_atif_internal(self) -> dict:
        """Convert current trajectory to ATIF format for incremental saving.

        Uses the existing convert_to_atif logic but with instance attributes.
        """
        return convert_to_atif(
            trajectory=self.trajectory,
            model_name=self.model,
            agent_name="minimal-codex",
            agent_version="0.3.0",
            session_id=self.session_id,
        )


def convert_to_atif(
    trajectory: list[dict],
    model_name: str,
    agent_version: str = "0.1.0",
    agent_name: str = "minimal-codex",
    session_id: str = None,
) -> dict:
    """Convert internal trajectory to ATIF v1.4 format for Harbor compatibility.

    Args:
        trajectory: Internal trajectory list
        model_name: Model name used
        agent_version: Version of this agent
        agent_name: Name of the agent (for subagent trajectories)
        session_id: Optional stable session ID (generates new one if not provided)

    Returns:
        ATIF-formatted trajectory dict
    """
    steps = []
    step_id = 1

    # Aggregate metrics
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_cost_usd = 0.0

    for entry in trajectory:
        timestamp = entry.get("timestamp", datetime.now(timezone.utc).isoformat())

        if entry["role"] == "system":
            steps.append({
                "step_id": step_id,
                "timestamp": timestamp,
                "source": "system",
                "message": entry.get("content", ""),
            })
            step_id += 1

        elif entry["role"] == "user":
            steps.append({
                "step_id": step_id,
                "timestamp": timestamp,
                "source": "user",
                "message": entry.get("content", ""),
            })
            step_id += 1

        elif entry["role"] == "assistant":
            # Build tool_calls list
            tool_calls_atif = None
            if entry.get("tool_calls"):
                tool_calls_atif = [
                    {
                        "tool_call_id": tc.get("id", f"call_{step_id}_{i}"),
                        "function_name": tc.get("function", {}).get("name", ""),
                        "arguments": json.loads(tc.get("function", {}).get("arguments", "{}")),
                    }
                    for i, tc in enumerate(entry.get("tool_calls", []))
                ]

            # Build metrics if available
            usage = entry.get("usage", {})
            metrics = None
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                cached_tokens = usage.get("cached_tokens", 0)
                cost_usd = usage.get("cost_usd")

                metrics = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
                if cached_tokens > 0:
                    metrics["cached_tokens"] = cached_tokens
                if cost_usd:
                    metrics["cost_usd"] = cost_usd

                # Accumulate totals
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_cached_tokens += cached_tokens
                if cost_usd:
                    total_cost_usd += cost_usd

            step = {
                "step_id": step_id,
                "timestamp": timestamp,
                "source": "agent",
                "model_name": model_name,
                "message": entry.get("content", ""),
            }
            if entry.get("reasoning"):
                step["reasoning_content"] = entry.get("reasoning")
            if tool_calls_atif:
                step["tool_calls"] = tool_calls_atif
            if metrics:
                step["metrics"] = metrics

            steps.append(step)
            step_id += 1

        elif entry["role"] == "tool":
            # Attach tool result as observation to previous agent step
            # CRITICAL: Must link via source_call_id for proper visualization
            if steps and steps[-1].get("source") == "agent":
                prev_step = steps[-1]
                if "observation" not in prev_step:
                    prev_step["observation"] = {"results": []}

                prev_step["observation"]["results"].append({
                    "source_call_id": entry.get("tool_call_id"),
                    "content": entry.get("content", ""),
                })

    # Build final trajectory
    final_metrics = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_steps": len([s for s in steps if s.get("source") == "agent"]),
    }
    if total_cached_tokens > 0:
        final_metrics["total_cached_tokens"] = total_cached_tokens
    if total_cost_usd > 0:
        final_metrics["total_cost_usd"] = total_cost_usd

    return {
        "schema_version": "ATIF-v1.4",
        "session_id": session_id or str(uuid.uuid4()),
        "agent": {
            "name": agent_name,
            "version": agent_version,
            "model_name": model_name,
            "extra": {"framework": "minimal-codex"}
        },
        "steps": steps,
        "final_metrics": final_metrics,
        "notes": "Generated by minimal-codex agent",
    }


def main():
    """CLI entry point for the Minimal Codex Agent."""
    parser = argparse.ArgumentParser(description="Minimal Codex Agent - autonomous coding agent")
    parser.add_argument("--task", required=True, help="Task to execute")
    parser.add_argument("--model", required=True, help="Model to use (via LiteLLM)")
    parser.add_argument("--cwd", default=".", help="Working directory")
    parser.add_argument("--output", help="Output file for results JSON")
    parser.add_argument("--trajectory", help="Output file for ATIF trajectory")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Maximum turns (default: unlimited, uses compaction like real Codex)")

    # Feature flags (all enabled by default, use --no-* to disable)
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable token streaming to stdout")
    parser.add_argument("--no-pty", action="store_true",
                        help="Disable PTY shell sessions (exec_command/write_stdin)")
    parser.add_argument("--no-web-search", action="store_true",
                        help="Disable web search tool")
    parser.add_argument("--no-plan-mode", action="store_true",
                        help="Disable autonomous planning workflow")
    parser.add_argument("--no-subagents", action="store_true",
                        help="Disable subagent invocation")

    # Plan mode activation (uses autonomous planning if enabled)
    parser.add_argument("--plan", action="store_true",
                        help="Use autonomous planning mode (research -> plan -> execute)")

    args = parser.parse_args()

    # Configure features based on CLI flags
    features = Features.from_env()  # Start with env-based config
    if args.no_stream:
        features.disable(Feature.STREAMING)
    if args.no_pty:
        features.disable(Feature.PTY_SHELL)
    if args.no_web_search:
        features.disable(Feature.WEB_SEARCH)
    if args.no_plan_mode:
        features.disable(Feature.PLAN_MODE)
    if args.no_subagents:
        features.disable(Feature.SUBAGENTS)

    # Run the agent
    agent = CodexAgent(model=args.model, cwd=args.cwd, features=features)
    result = agent.run(
        args.task,
        max_turns=args.max_turns,
        use_plan_mode=args.plan,
        trajectory_path=args.trajectory,
        output_path=args.output,
    )

    # Build output
    output = {
        "output": result.get("output"),
        "error": result.get("error"),
        "usage": result.get("usage", {
            "input_tokens": result.get("input_tokens", 0),
            "cached_input_tokens": result.get("cached_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
        })
    }

    # Add plan mode artifacts if present
    if result.get("plan_file"):
        output["plan_file"] = result["plan_file"]
        print(f"[Plan Mode] Plan file: {result['plan_file']}")
    if result.get("subagent_trajectories"):
        output["subagent_trajectories"] = result["subagent_trajectories"]
        print(f"[Plan Mode] Subagent trajectories: {len(result['subagent_trajectories'])} files")
        for traj in result["subagent_trajectories"]:
            print(f"  - {traj}")

    # Write output JSON
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2))

    # Write ATIF trajectory
    if args.trajectory:
        atif = convert_to_atif(result["trajectory"], args.model)
        Path(args.trajectory).parent.mkdir(parents=True, exist_ok=True)
        Path(args.trajectory).write_text(json.dumps(atif, indent=2))

    # Print output for capture
    print(json.dumps(output))


if __name__ == "__main__":
    main()
