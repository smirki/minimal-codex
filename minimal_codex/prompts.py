"""System prompts for the Minimal Codex Agent.

Loads the exact Codex prompt from prompt.md for Terminal-Bench compatibility.
"""

from pathlib import Path

# Load prompt from file at module load time
_PROMPT_FILE = Path(__file__).parent / "prompt.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8")

# Environment context template (matching Codex's format)
ENVIRONMENT_CONTEXT_TEMPLATE = '''<environment_context>
  <cwd>{cwd}</cwd>
  <approval_policy>never</approval_policy>
  <sandbox_mode>danger-full-access</sandbox_mode>
  <network_access>enabled</network_access>
  <shell>bash</shell>
</environment_context>'''


def get_system_prompt(model_name: str = "") -> str:
    """Get the system prompt.

    Args:
        model_name: Model name (unused, kept for API compatibility)

    Returns:
        System prompt string (exact Codex prompt from prompt.md)
    """
    return SYSTEM_PROMPT


def get_environment_context(cwd: str = "/app") -> str:
    """Get the environment context message.

    Args:
        cwd: Current working directory

    Returns:
        Environment context XML string
    """
    return ENVIRONMENT_CONTEXT_TEMPLATE.format(cwd=cwd)
