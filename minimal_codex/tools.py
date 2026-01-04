"""Tool definitions for the Minimal Codex Agent.

These match Codex CLI's exact tool specifications.
"""

import subprocess
import re
import fnmatch
from pathlib import Path
from typing import Any

# Tool definitions in OpenAI function calling format

SHELL_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "shell_command",
        "description": "Runs a shell command and returns output. Use rg (ripgrep) for fast searching.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell script to execute"
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory (optional)"
                },
                "timeout_ms": {
                    "type": "number",
                    "description": "Timeout in milliseconds (default 120000)"
                }
            },
            "required": ["command"],
            "additionalProperties": False
        }
    }
}

APPLY_PATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "apply_patch",
        "description": """Edit files using patch format. This is a FREEFORM tool - provide the patch directly.

Format:
*** Begin Patch
*** Add File: path/to/new.py
+line1
+line2
*** Delete File: path/to/old.py
*** Update File: path/to/file.py
*** Move to: path/to/newname.py
@@ def function_name():
 context line (unchanged)
-line to remove
+line to add
*** End of File
*** End Patch

Rules:
- Lines with + are additions
- Lines with - are removals
- Lines with space are context (help locate the change)
- @@ markers show context/location hints
- *** End of File marks EOF-relative changes""",
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "The patch content"
                }
            },
            "required": ["patch"],
            "additionalProperties": False
        }
    }
}

UPDATE_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "update_plan",
        "description": "Updates the task plan. At most one step can be in_progress at a time. Do not use for simple tasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Optional explanation of plan changes"
                },
                "plan": {
                    "type": "array",
                    "description": "List of plan steps (5-7 words each)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                        },
                        "required": ["step", "status"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["plan"],
            "additionalProperties": False
        }
    }
}

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read file contents with 1-indexed line numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "number", "description": "Start line (1-indexed)"},
                "limit": {"type": "number", "description": "Max lines to read"}
            },
            "required": ["file_path"],
            "additionalProperties": False
        }
    }
}

LIST_DIR_TOOL = {
    "type": "function",
    "function": {
        "name": "list_dir",
        "description": "List directory entries with depth control",
        "parameters": {
            "type": "object",
            "properties": {
                "dir_path": {"type": "string"},
                "depth": {"type": "number", "description": "Max depth (default 1)"},
                "limit": {"type": "number", "description": "Max entries"}
            },
            "required": ["dir_path"],
            "additionalProperties": False
        }
    }
}

GREP_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "grep_files",
        "description": "Find files matching regex pattern, sorted by modification time",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern"},
                "path": {"type": "string", "description": "Directory to search"},
                "include": {"type": "string", "description": "Glob pattern for files"},
                "limit": {"type": "number", "description": "Max results"}
            },
            "required": ["pattern"],
            "additionalProperties": False
        }
    }
}

# All tools combined
TOOLS = [
    SHELL_COMMAND_TOOL,
    APPLY_PATCH_TOOL,
    UPDATE_PLAN_TOOL,
    READ_FILE_TOOL,
    LIST_DIR_TOOL,
    GREP_FILES_TOOL,
]

# Tools that can run in parallel (read-only or idempotent)
PARALLEL_TOOLS = {"update_plan", "read_file", "list_dir", "grep_files"}
# Tools that need exclusive access (side effects)
SEQUENTIAL_TOOLS = {"shell_command", "apply_patch"}

# Truncation settings
MAX_TOOL_OUTPUT_BYTES = 10000


def truncate_output(output: str, max_bytes: int = MAX_TOOL_OUTPUT_BYTES) -> str:
    """Truncate output preserving beginning and end (Codex style)."""
    if len(output) <= max_bytes:
        return output

    # Split budget: 40% beginning, 40% end, 20% for marker
    left_budget = int(max_bytes * 0.4)
    right_budget = int(max_bytes * 0.4)

    left = output[:left_budget]
    right = output[-right_budget:]
    omitted = len(output) - left_budget - right_budget

    return f"{left}\n\n... [{omitted} characters omitted] ...\n\n{right}"


def execute_shell(args: dict, cwd: Path) -> str:
    """Execute shell command."""
    command = args["command"]
    workdir = args.get("workdir", str(cwd))
    timeout_ms = args.get("timeout_ms", 120000)
    timeout_sec = timeout_ms / 1000

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"stderr: {result.stderr}")
        output_parts.append(f"exit code: {result.returncode}")

        return "\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout_sec}s"
    except Exception as e:
        return f"Error: {str(e)}"


def read_file(args: dict, cwd: Path) -> str:
    """Read file contents with line numbers."""
    file_path = args["file_path"]

    # Handle absolute and relative paths
    if Path(file_path).is_absolute():
        path = Path(file_path)
    else:
        path = cwd / file_path

    if not path.exists():
        return f"Error: File not found: {file_path}"

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").split('\n')
    except Exception as e:
        return f"Error reading file: {e}"

    offset = args.get("offset", 1) - 1  # Convert to 0-indexed
    limit = args.get("limit", len(lines))

    result_lines = []
    for i, line in enumerate(lines[offset:offset+limit], start=offset+1):
        result_lines.append(f"{i:6d}\t{line}")

    return '\n'.join(result_lines)


def list_dir(args: dict, cwd: Path) -> str:
    """List directory entries."""
    dir_path = args["dir_path"]

    # Handle absolute and relative paths
    if Path(dir_path).is_absolute():
        path = Path(dir_path)
    else:
        path = cwd / dir_path

    if not path.exists():
        return f"Error: Directory not found: {dir_path}"
    if not path.is_dir():
        return f"Error: Not a directory: {dir_path}"

    depth = args.get("depth", 1)
    limit = args.get("limit", 100)

    entries = []

    def walk(p: Path, current_depth: int):
        if current_depth > depth or len(entries) >= limit:
            return
        try:
            for item in sorted(p.iterdir()):
                if len(entries) >= limit:
                    break
                try:
                    rel = item.relative_to(cwd)
                except ValueError:
                    rel = item
                entries.append(str(rel) + ("/" if item.is_dir() else ""))
                if item.is_dir() and current_depth < depth:
                    walk(item, current_depth + 1)
        except PermissionError:
            pass

    walk(path, 1)
    return '\n'.join(entries) if entries else "(empty directory)"


def grep_files(args: dict, cwd: Path) -> str:
    """Search files using ripgrep or fallback."""
    pattern = args["pattern"]
    search_path = cwd / args.get("path", ".")
    include = args.get("include", "*")
    limit = args.get("limit", 50)

    # Try ripgrep first (faster)
    try:
        cmd = ["rg", "-l", "--glob", include, pattern, str(search_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            files = [f for f in result.stdout.strip().split('\n') if f][:limit]
            return '\n'.join(files) if files else "No matches found"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback to Python regex
    matches = []
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    for f in search_path.rglob("*"):
        if len(matches) >= limit:
            break
        if f.is_file() and fnmatch.fnmatch(f.name, include):
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if regex.search(content):
                    try:
                        matches.append(str(f.relative_to(cwd)))
                    except ValueError:
                        matches.append(str(f))
            except Exception:
                pass

    return '\n'.join(matches) if matches else "No matches found"


def update_plan(args: dict, current_plan: list) -> tuple[str, list]:
    """Update the task plan. Returns (message, updated_plan)."""
    new_plan = args["plan"]
    explanation = args.get("explanation", "")

    # Validate: at most one in_progress
    in_progress = [s for s in new_plan if s["status"] == "in_progress"]
    if len(in_progress) > 1:
        return "Warning: Multiple steps marked in_progress", new_plan

    return (f"Plan updated. {explanation}" if explanation else "Plan updated."), new_plan
