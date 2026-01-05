"""Template variable resolver for prompts.

Resolves ${TOOL_NAME} style variables at runtime for Windows compatibility.
"""

from pathlib import Path
from typing import Optional

# Tool name mappings for Minimal Codex
TEMPLATE_VARS = {
    "${GLOB_TOOL_NAME}": "list_dir",
    "${GREP_TOOL_NAME}": "grep_files",
    "${READ_TOOL_NAME}": "read_file",
    "${BASH_TOOL_NAME}": "shell_command",
    "${EDIT_TOOL_NAME}": "apply_patch",
    "${WRITE_TOOL_NAME}": "apply_patch",
    "${UPDATE_PLAN_TOOL_NAME}": "update_plan",
    "${INVOKE_SUBAGENT_TOOL_NAME}": "invoke_subagent",
    "${SAVE_PLAN_TOOL_NAME}": "save_plan",
    "${WEB_SEARCH_TOOL_NAME}": "web_search",
    # Agent name
    "${AGENT_NAME}": "Minimal Codex",
}


def resolve_template(content: str, extra_vars: Optional[dict] = None) -> str:
    """Resolve template variables in content.

    Args:
        content: Template content with ${VAR} placeholders
        extra_vars: Additional variables to resolve

    Returns:
        Content with variables resolved
    """
    result = content

    # Apply standard vars
    for var, value in TEMPLATE_VARS.items():
        result = result.replace(var, value)

    # Apply extra vars if provided
    if extra_vars:
        for var, value in extra_vars.items():
            result = result.replace(var, value)

    return result


def load_prompt_template(prompt_name: str, prompts_dir: Optional[Path] = None) -> str:
    """Load and resolve a prompt template.

    Args:
        prompt_name: Name of prompt file (without .md extension)
        prompts_dir: Directory containing prompts (defaults to package prompts/)

    Returns:
        Resolved prompt content
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent / "prompts"

    path = prompts_dir / f"{prompt_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    content = path.read_text(encoding="utf-8")
    return resolve_template(content)
