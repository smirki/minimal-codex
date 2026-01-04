"""Main agent class for the Minimal Codex Agent.

Implements the core agent loop matching Codex CLI's logic:
- API calls with retry and backoff
- Tool execution (parallel/sequential)
- Trajectory recording for ATIF format
- Conversation management
"""

import argparse
import json
import os
import random
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
from .prompts import get_system_prompt
from .tools import (
    TOOLS,
    PARALLEL_TOOLS,
    execute_shell,
    read_file,
    list_dir,
    grep_files,
    update_plan,
    truncate_output,
)

# LiteLLM configuration - important for compatibility with various APIs
litellm.drop_params = True

# ============================================================================
# Context Compaction Constants (from Codex's compact.rs and truncate.rs)
# ============================================================================

# Exact constant from Codex's truncate.rs
APPROX_BYTES_PER_TOKEN = 4
COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000
COMPACT_USER_MESSAGE_MAX_BYTES = COMPACT_USER_MESSAGE_MAX_TOKENS * APPROX_BYTES_PER_TOKEN  # 80,000

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


def collect_user_messages(
    messages: list[dict],
    max_bytes: int = COMPACT_USER_MESSAGE_MAX_BYTES
) -> str:
    """Extract user messages from history up to max_bytes.

    This matches Codex's collect_user_messages() exactly:
    - Iterates through messages in order
    - Concatenates user message content
    - Stops when byte limit is reached
    """
    result = []
    total_bytes = 0

    for msg in messages:
        if msg.get("role") != "user":
            continue

        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle multi-part messages (text + images)
            content = " ".join(
                part.get("text", "") for part in content
                if part.get("type") == "text"
            )

        content_bytes = len(content.encode('utf-8'))

        if total_bytes + content_bytes > max_bytes:
            # Truncate to fit
            remaining = max_bytes - total_bytes
            if remaining > 0:
                # Simple byte truncation (Codex does UTF-8 safe truncation)
                truncated = content.encode('utf-8')[:remaining].decode('utf-8', errors='ignore')
                result.append(truncated)
            break

        result.append(content)
        total_bytes += content_bytes

    return "\n\n".join(result)


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

    def __init__(self, model: str, cwd: str = ".", context_window: int = 128000):
        """Initialize the agent.

        Args:
            model: Model name to use (passed to LiteLLM)
            cwd: Working directory for the agent
            context_window: Model's context window size (for compaction threshold)
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

        # Context compaction thresholds (from Codex config)
        self.context_window = context_window
        self.compaction_threshold = 0.8  # Compact at 80% of context
        self.last_compaction_tokens = 0

    def run(self, task: str, max_turns: int = 100) -> dict:
        """Run the agent until task completion.

        Args:
            task: The task to perform
            max_turns: Maximum number of turns before stopping

        Returns:
            Dict with output, trajectory, token counts, etc.
        """
        # Initialize conversation with context
        self.messages = build_initial_messages(self.cwd, task)

        # Record initial messages to trajectory
        for msg in self.messages:
            self._record_message(msg["role"], msg["content"])

        for turn in range(max_turns):
            # Check for compaction before API call (from Codex's compact.rs)
            if self.compact_conversation():
                print(f"[Compacted context at turn {turn}]")

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
            self.messages.append(assistant_message.model_dump())

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

    def _call_api_with_retry(self, max_retries: int = 3) -> Any:
        """Call API with exponential backoff retry.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            API response object
        """
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                response = completion(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    parallel_tool_calls=True,
                )
                return response
            except Exception as e:
                if attempt == max_retries:
                    raise

                # Check if retryable
                if not self._is_retryable_error(e):
                    raise

                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt)
                jitter = random.uniform(0.9, 1.1)
                time.sleep(delay * jitter)

        raise Exception("Max retries exceeded")

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error should trigger retry."""
        error_str = str(error).lower()
        return any(x in error_str for x in ['429', '500', '502', '503', 'timeout', 'connection'])

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
        """Perform context compaction using LLM summarization.

        Returns True if compaction was performed.
        """
        if not self.should_compact():
            return False

        # Build prompt for summarization
        history_text = self._format_history_for_summary()

        summary_messages = [
            {"role": "system", "content": COMPACT_PROMPT},
            {"role": "user", "content": history_text}
        ]

        try:
            # Call model for summarization
            response = completion(
                model=self.model,
                messages=summary_messages,
                max_tokens=4000,  # Allow substantial summary
            )
            summary = response.choices[0].message.content

            # Track tokens used for summary
            if hasattr(response, 'usage') and response.usage:
                self.total_input_tokens += getattr(response.usage, 'prompt_tokens', 0)
                self.total_output_tokens += getattr(response.usage, 'completion_tokens', 0)

            # Build new compacted history
            self.messages = build_compacted_history(self.messages, summary)
            self.last_compaction_tokens = self._estimate_context_tokens()

            return True

        except Exception as e:
            # Log error but don't fail - just continue with full history
            print(f"Warning: Compaction failed: {e}")
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
        """Execute tool calls, parallelizing where supported.

        Args:
            tool_calls: List of tool call objects from the API

        Returns:
            List of tool result dicts
        """
        results = []

        # Separate parallel vs sequential tools
        parallel_calls = []
        sequential_calls = []

        for tc in tool_calls:
            if tc.function.name in PARALLEL_TOOLS:
                parallel_calls.append(tc)
            else:
                sequential_calls.append(tc)

        # Execute sequential tools first (shell, apply_patch need exclusive access)
        for tc in sequential_calls:
            result = self._execute_single_tool(tc)
            results.append(result)

        # Execute parallel tools concurrently
        if parallel_calls:
            with ThreadPoolExecutor(max_workers=len(parallel_calls)) as executor:
                futures = [executor.submit(self._execute_single_tool, tc) for tc in parallel_calls]
                for f in futures:
                    results.append(f.result())

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

    def _record_message(self, role: str, content: str, **kwargs):
        """Record a message with timestamp for trajectory."""
        self.trajectory.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        })

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

    def _record_tool_result(self, tool_call_id: str, content: str):
        """Record tool execution result."""
        self.trajectory.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


def convert_to_atif(
    trajectory: list[dict],
    model_name: str,
    agent_version: str = "0.1.0"
) -> dict:
    """Convert internal trajectory to ATIF v1.4 format for Harbor compatibility.

    Args:
        trajectory: Internal trajectory list
        model_name: Model name used
        agent_version: Version of this agent

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
        "session_id": str(uuid.uuid4()),
        "agent": {
            "name": "minimal-codex",
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
    parser.add_argument("--max-turns", type=int, default=100, help="Maximum turns")

    args = parser.parse_args()

    # Run the agent
    agent = CodexAgent(model=args.model, cwd=args.cwd)
    result = agent.run(args.task, max_turns=args.max_turns)

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
