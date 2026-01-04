"""Context building for the Minimal Codex Agent.

Handles:
- AGENTS.md discovery and loading (git root to CWD)
- Environment context XML generation
- Initial conversation message building
"""

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

from .prompts import get_system_prompt


def find_git_root(cwd: Path) -> Optional[Path]:
    """Find the git repository root from cwd.

    Args:
        cwd: Current working directory

    Returns:
        Path to git root, or None if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def discover_agents_md(cwd: Path) -> list[Path]:
    """Discover AGENTS.md files from git root to CWD.

    Walks from git root (or cwd if not in a repo) to cwd,
    collecting AGENTS.md or AGENTS.override.md files.
    Override files take precedence.

    Args:
        cwd: Current working directory

    Returns:
        List of AGENTS.md file paths in order (root to cwd)
    """
    git_root = find_git_root(cwd)
    start = git_root if git_root else cwd

    paths = []

    # Walk from start to cwd
    try:
        relative = cwd.relative_to(start)
        parts = relative.parts
    except ValueError:
        # cwd is not under git_root, just check cwd
        parts = []

    # Check start directory
    current = start
    for filename in ["AGENTS.override.md", "AGENTS.md"]:
        agent_file = current / filename
        if agent_file.exists():
            paths.append(agent_file)
            break

    # Walk down to cwd
    for part in parts:
        current = current / part
        for filename in ["AGENTS.override.md", "AGENTS.md"]:
            agent_file = current / filename
            if agent_file.exists():
                paths.append(agent_file)
                break

    return paths


def load_agents_md(cwd: Path, max_bytes: int = 50000) -> str:
    """Load and concatenate AGENTS.md content.

    Args:
        cwd: Current working directory
        max_bytes: Maximum total bytes to include

    Returns:
        Concatenated AGENTS.md content
    """
    paths = discover_agents_md(cwd)
    contents = []
    total_bytes = 0

    for path in paths:
        try:
            content = path.read_text(encoding='utf-8')
            if total_bytes + len(content) > max_bytes:
                # Truncate this content to fit
                remaining = max_bytes - total_bytes
                if remaining > 100:  # Only add if meaningful amount
                    content = content[:remaining] + "\n... (truncated)"
                    contents.append(f"# AGENTS.md from {path.parent}\n\n{content}")
                break
            contents.append(f"# AGENTS.md from {path.parent}\n\n{content}")
            total_bytes += len(content)
        except Exception:
            pass

    return "\n\n--- project-doc ---\n\n".join(contents) if contents else ""


def detect_shell() -> str:
    """Detect the current shell."""
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return "zsh"
    elif "bash" in shell:
        return "bash"
    elif "fish" in shell:
        return "fish"
    elif platform.system() == "Windows":
        return "powershell"
    return "bash"


def build_environment_context(cwd: Path) -> str:
    """Build environment context in XML format (matches Codex).

    Args:
        cwd: Current working directory

    Returns:
        Environment context XML string
    """
    shell = detect_shell()
    system = platform.system()

    return f'''<environment_context>
  <cwd>{cwd}</cwd>
  <platform>{system}</platform>
  <approval_policy>never</approval_policy>
  <sandbox_mode>danger-full-access</sandbox_mode>
  <network_access>enabled</network_access>
  <shell>{shell}</shell>
</environment_context>'''


def build_initial_messages(cwd: Path, task: str, system_prompt: Optional[str] = None) -> list[dict]:
    """Build the initial conversation messages.

    Args:
        cwd: Current working directory
        task: The task to perform
        system_prompt: Optional custom system prompt

    Returns:
        List of message dicts for the conversation
    """
    messages = []

    # 1. System message with base instructions
    if system_prompt is None:
        system_prompt = get_system_prompt()

    messages.append({
        "role": "system",
        "content": system_prompt
    })

    # 2. User message with AGENTS.md instructions (if any)
    agents_md = load_agents_md(cwd)
    if agents_md:
        messages.append({
            "role": "user",
            "content": f"# AGENTS.md instructions for {cwd}\n\n<INSTRUCTIONS>\n{agents_md}\n</INSTRUCTIONS>"
        })

    # 3. User message with environment context
    env_context = build_environment_context(cwd)
    messages.append({
        "role": "user",
        "content": env_context
    })

    # 4. The actual task
    messages.append({
        "role": "user",
        "content": task
    })

    return messages
