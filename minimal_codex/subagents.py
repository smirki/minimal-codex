"""Subagent system for Minimal Codex.

Provides:
- SubagentConfig: Configuration for a subagent
- SubagentSession: Resumable subagent session state
- SubagentManager: Manages subagent configurations and invocation

Architecture:
- MAIN AGENT is the orchestrator
- Subagents are workers invoked BY the main agent
- Subagents CANNOT invoke other subagents (no nested delegation)
"""

import uuid
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from litellm import completion
import yaml

from .prompt_templates import load_prompt_template


@dataclass
class SubagentConfig:
    """Configuration for a subagent."""
    name: str
    description: str
    tools: Optional[list[str]]  # None = all tools
    model: str  # "inherit" or specific model name
    system_prompt: str


@dataclass
class SubagentSession:
    """Resumable subagent session state."""
    agent_id: str
    name: str
    messages: list[dict]

    def save(self, transcripts_dir: Path):
        """Save session to transcript file."""
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        path = transcripts_dir / f"agent-{self.agent_id}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for msg in self.messages:
                f.write(json.dumps(msg) + "\n")

    @classmethod
    def load(cls, agent_id: str, transcripts_dir: Path) -> Optional["SubagentSession"]:
        """Load session from transcript file."""
        path = transcripts_dir / f"agent-{agent_id}.jsonl"
        if not path.exists():
            return None
        messages = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    messages.append(json.loads(line))
        return cls(agent_id=agent_id, name="", messages=messages)


def get_builtin_subagents() -> dict[str, SubagentConfig]:
    """Get built-in subagent configurations.

    All prompts are loaded from prompts/subagent/ folder to ensure:
    - Subagent framing ("You are a SUBAGENT, return quickly")
    - Early stopping instructions
    - Context awareness
    """
    return {
        "general-purpose": SubagentConfig(
            name="general-purpose",
            description="General-purpose agent for complex multi-step tasks, code search, and research",
            tools=None,  # All tools
            model="inherit",
            system_prompt=load_prompt_template("subagent/general_purpose"),
        ),
        "Plan": SubagentConfig(
            name="Plan",
            description="Read-only exploration for creating implementation plans",
            tools=["read_file", "list_dir", "grep_files", "shell_command"],  # Read-only tools only
            model="inherit",
            system_prompt=load_prompt_template("subagent/plan"),
        ),
        "Explore": SubagentConfig(
            name="Explore",
            description="Fast codebase exploration - find files, search code, answer questions",
            tools=["read_file", "list_dir", "grep_files"],
            model="inherit",
            system_prompt=load_prompt_template("subagent/explore"),
        ),
    }


# Lazy-loaded to ensure prompt_templates is available
_BUILTIN_SUBAGENTS: Optional[dict[str, SubagentConfig]] = None


def get_builtin_subagent_configs() -> dict[str, SubagentConfig]:
    """Get or initialize built-in subagent configs."""
    global _BUILTIN_SUBAGENTS
    if _BUILTIN_SUBAGENTS is None:
        _BUILTIN_SUBAGENTS = get_builtin_subagents()
    return _BUILTIN_SUBAGENTS


class SubagentManager:
    """Manages subagent configurations and invocation.

    The SubagentManager is used by the MAIN AGENT to:
    - Load built-in and custom subagent configurations
    - Invoke subagents for specialized tasks
    - Handle parallel subagent execution
    - Manage resumable sessions

    IMPORTANT: Subagents CANNOT invoke other subagents.
    The invoke_subagent tool is removed from subagent tool lists.
    """

    MAX_CONCURRENT = 10  # Max parallel subagents

    def __init__(self, cwd: Path, model: str, all_tools: list):
        """Initialize the SubagentManager.

        Args:
            cwd: Current working directory
            model: Model name to use for subagents with "inherit"
            all_tools: List of all tool definitions
        """
        self.cwd = cwd
        self.model = model
        self.all_tools = all_tools
        self.configs: dict[str, SubagentConfig] = {}
        self.transcripts_dir = cwd / ".tessa" / "transcripts"

        # Load built-in subagents
        self.configs.update(get_builtin_subagent_configs())

        # Load custom subagents from .tessa/agents/
        self._load_custom_subagents()

    def _load_custom_subagents(self):
        """Load subagent configs from .tessa/agents/*.md"""
        agents_dir = self.cwd / ".tessa" / "agents"
        if not agents_dir.exists():
            return

        for md_file in agents_dir.glob("*.md"):
            config = self._parse_subagent_file(md_file)
            if config:
                self.configs[config.name] = config

    def _parse_subagent_file(self, path: Path) -> Optional[SubagentConfig]:
        """Parse a subagent markdown file with YAML frontmatter."""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return None

        # Split frontmatter and body
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1])
                except yaml.YAMLError:
                    return None

                system_prompt = parts[2].strip()

                return SubagentConfig(
                    name=frontmatter.get("name", path.stem),
                    description=frontmatter.get("description", ""),
                    tools=self._parse_tools(frontmatter.get("tools")),
                    model=frontmatter.get("model", "inherit"),
                    system_prompt=system_prompt,
                )
        return None

    def _parse_tools(self, tools_str: Optional[str]) -> Optional[list[str]]:
        """Parse tools string (comma-separated or None for all)."""
        if tools_str is None:
            return None
        if isinstance(tools_str, list):
            return tools_str
        return [t.strip() for t in tools_str.split(",")]

    def get_available_subagents(self) -> list[dict]:
        """Get list of available subagents for the tool description."""
        return [
            {"name": c.name, "description": c.description}
            for c in self.configs.values()
        ]

    def invoke(
        self,
        name: str,
        task: str,
        max_turns: int = 10,
        resume_id: Optional[str] = None,
        context: Optional[str] = None,
    ) -> tuple[str, str, "SubagentSession"]:
        """Invoke a subagent by name.

        Args:
            name: Subagent name
            task: Task description
            max_turns: Max turns before returning (subagent should stop earlier when done)
            resume_id: Optional agent_id to resume previous session
            context: Optional context/findings from main agent to share

        Returns:
            (result, agent_id, session) - Result text, agent_id, and full session for trajectory
        """
        if name not in self.configs:
            available = list(self.configs.keys())
            return f"Error: Unknown subagent '{name}'. Available: {available}", "", None

        config = self.configs[name]

        # Resolve model
        model = self.model if config.model == "inherit" else config.model

        # Resolve tools - IMPORTANT: Remove invoke_subagent to prevent nesting
        tools = self._get_tools_for_subagent(config.tools, exclude_subagent=True)

        # Build task with context prefix if provided
        full_task = task
        if context:
            full_task = f"""## Context from Main Agent
{context}

## Your Task
{task}"""

        # Load or create session
        agent_id = resume_id or str(uuid.uuid4())[:8]
        session = None
        if resume_id:
            session = SubagentSession.load(resume_id, self.transcripts_dir)

        # Run subagent loop
        result, final_session = self._run_subagent(
            config, full_task, model, tools, max_turns, session, agent_id
        )

        # Save session for potential resumption
        final_session.save(self.transcripts_dir)

        return result, agent_id, final_session

    def _get_tools_for_subagent(
        self,
        tool_names: Optional[list[str]],
        exclude_subagent: bool = True,
    ) -> list:
        """Get tool definitions for subagent.

        Args:
            tool_names: List of tool names to include (None = all)
            exclude_subagent: Remove invoke_subagent to prevent nesting

        Returns:
            List of tool definitions
        """
        tools = self.all_tools.copy()

        if tool_names is not None:
            # Filter to requested tools
            tools = [t for t in tools if t["function"]["name"] in tool_names]

        if exclude_subagent:
            # Remove invoke_subagent to prevent nesting
            tools = [t for t in tools if t["function"]["name"] != "invoke_subagent"]

        return tools

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

    def _run_subagent(
        self,
        config: SubagentConfig,
        task: str,
        model: str,
        tools: list,
        max_turns: int,
        session: Optional[SubagentSession],
        agent_id: str,
    ) -> tuple[str, SubagentSession]:
        """Execute subagent loop and return findings.

        Args:
            config: Subagent configuration
            task: Task description
            model: Model to use
            tools: Tool definitions
            max_turns: Max turns
            session: Optional existing session to resume
            agent_id: Agent ID for this session

        Returns:
            (result, session) - Result text and final session state
        """

        # Initialize or resume messages
        if session and session.messages:
            messages = session.messages.copy()
            # Add new task as continuation
            messages.append({"role": "user", "content": f"[Continuation] {task}"})
        else:
            messages = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": task},
            ]

        for _ in range(max_turns):
            try:
                response = completion(
                    model=model,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                )
            except Exception as e:
                return f"Error calling LLM: {str(e)}", SubagentSession(agent_id, config.name, messages)

            message = response.choices[0].message
            # Use explicit serialization per Codex pattern (includes "type": "function")
            messages.append(self._serialize_assistant_message(message))

            # If no tool calls, subagent is done
            if not message.tool_calls:
                result = message.content or "(No response)"
                return result, SubagentSession(agent_id, config.name, messages)

            # Execute tools
            for tc in message.tool_calls:
                tool_result = self._execute_tool(tc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

        # Max turns reached
        return "(Max turns reached)", SubagentSession(agent_id, config.name, messages)

    def _execute_tool(self, tool_call) -> str:
        """Execute a tool call for subagent.

        Delegates to tool implementations in tools.py.
        """
        from .tools import execute_tool
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON in tool arguments"

        return execute_tool(
            tool_call.function.name,
            args,
            self.cwd
        )
