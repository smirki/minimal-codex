# Minimal Codex Agent

A 1:1 replica of Codex CLI's autonomous agent logic

## Features

- Exact match of Codex CLI's `apply_patch` with 4-level fuzzy matching (`seek_sequence`)
- ATIF v1.4 trajectory format support for Harbor visualization
- LiteLLM integration for any OpenAI-compatible API
- Parallel tool execution where supported
- AGENTS.md discovery and loading

## Installation

```bash
pip install git+https://github.com/YOUR_USERNAME/minimal-codex.git
```

Or with uv:
```bash
uv tool install git+https://github.com/YOUR_USERNAME/minimal-codex.git
```

## Usage

```bash
minimal-codex --task "Your task here" --model "model-name" --output results.json --trajectory trajectory.json
```

## Tools

- `shell_command` - Run shell commands
- `apply_patch` - Edit files using Codex's exact patch format
- `read_file` - Read file contents with line numbers
- `list_dir` - List directory entries
- `grep_files` - Search files using regex
- `update_plan` - Track multi-step task progress

## License

MIT
