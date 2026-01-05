"""Plan file management for autonomous planning mode.

Handles:
- Generating plan file names (word_word_word.md)
- Saving plans to .tessa/plans/
- Loading plans from file
- Plan format with steps, status, critical files
"""

import random
from pathlib import Path
from datetime import datetime
from typing import Optional


# Word list for generating plan names
WORDS = [
    "swift", "brave", "calm", "dark", "eager", "fair", "gold", "happy",
    "iron", "jade", "keen", "lush", "mist", "noble", "oak", "pale",
    "quick", "rust", "sage", "true", "vast", "warm", "xenon", "young", "zeal",
    "alpha", "beta", "gamma", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "oscar", "papa", "quebec",
    "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee",
]


def generate_plan_name() -> str:
    """Generate a plan name like word_word_word.md"""
    words = random.sample(WORDS, 3)
    return f"{'_'.join(words)}.md"


class PlanManager:
    """Manages plan files for autonomous planning mode.

    Plans are stored in .tessa/plans/ with names like word_word_word.md.
    This makes them easy to identify and reference.
    """

    def __init__(self, base_dir: Path):
        """Initialize the PlanManager.

        Args:
            base_dir: Base directory (typically the working directory)
        """
        self.plans_dir = base_dir / ".tessa" / "plans"
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.current_plan_path: Optional[Path] = None

    def create_plan(
        self,
        task: str,
        steps: list,
        critical_files: list,
    ) -> Path:
        """Create and save a new plan file.

        Args:
            task: Original task description
            steps: List of {"step": str, "status": str}
            critical_files: List of file paths critical for implementation

        Returns:
            Path to the created plan file
        """
        name = generate_plan_name()
        path = self.plans_dir / name

        # Ensure unique name
        while path.exists():
            name = generate_plan_name()
            path = self.plans_dir / name

        plan_content = f"""# Plan: {name.replace('.md', '').replace('_', ' ').title()}

## Task
{task}

## Created
{datetime.now().isoformat()}

## Steps

{self._format_steps(steps)}

## Critical Files

{self._format_files(critical_files)}

## Progress Notes

(Updated as execution progresses)
"""

        path.write_text(plan_content, encoding="utf-8")
        self.current_plan_path = path
        return path

    def _format_steps(self, steps: list) -> str:
        """Format steps for markdown."""
        lines = []
        for i, step in enumerate(steps, 1):
            status = step.get("status", "pending")
            if status == "completed":
                symbol = "✓"
            elif status == "in_progress":
                symbol = "▶"
            else:
                symbol = "□"
            lines.append(f"{i}. [{symbol}] {step['step']}")
        return "\n".join(lines)

    def _format_files(self, files: list) -> str:
        """Format critical files for markdown."""
        return "\n".join(f"- `{f}`" for f in files)

    def load_plan(self, path: Path) -> dict:
        """Load a plan from file.

        Returns dict with: task, steps, critical_files, created
        """
        content = path.read_text(encoding="utf-8")
        return self._parse_plan(content)

    def _parse_plan(self, content: str) -> dict:
        """Parse plan markdown into structured data."""
        sections = {}
        current_section = None
        current_content = []

        for line in content.split("\n"):
            if line.startswith("## "):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = line[3:].strip().lower()
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        # Parse steps from markdown
        steps = []
        if "steps" in sections:
            for line in sections["steps"].split("\n"):
                line = line.strip()
                if line and line[0].isdigit() and "." in line:
                    # Extract step text and status
                    if "[✓]" in line:
                        status = "completed"
                    elif "[▶]" in line:
                        status = "in_progress"
                    else:
                        status = "pending"

                    # Extract step text after the symbol
                    if "]" in line:
                        step_text = line.split("]", 1)[-1].strip()
                    else:
                        step_text = line.split(".", 1)[-1].strip()

                    if step_text:
                        steps.append({"step": step_text, "status": status})

        # Parse critical files
        critical_files = []
        if "critical files" in sections:
            for line in sections["critical files"].split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    # Extract file path, removing backticks
                    file_path = line[2:].strip().strip("`")
                    if file_path:
                        critical_files.append(file_path)

        return {
            "task": sections.get("task", ""),
            "steps": steps,
            "critical_files": critical_files,
            "created": sections.get("created", ""),
        }

    def update_plan_step(self, path: Path, step_index: int, new_status: str):
        """Update a step's status in the plan file.

        Args:
            path: Path to the plan file
            step_index: 0-indexed step number to update
            new_status: New status ("pending", "in_progress", "completed")
        """
        plan = self.load_plan(path)
        if 0 <= step_index < len(plan["steps"]):
            plan["steps"][step_index]["status"] = new_status
            self._save_plan(path, plan)

    def _save_plan(self, path: Path, plan: dict):
        """Save plan dict back to file."""
        content = f"""# Plan: {path.stem.replace('_', ' ').title()}

## Task
{plan['task']}

## Created
{plan['created']}

## Steps

{self._format_steps(plan['steps'])}

## Critical Files

{self._format_files(plan['critical_files'])}

## Progress Notes

(Updated as execution progresses)
"""
        path.write_text(content, encoding="utf-8")

    def get_plan_context(self, path: Path) -> str:
        """Get plan content formatted for injection into context.

        This is used when executing a plan to inject it into the
        agent's context window.

        Args:
            path: Path to the plan file

        Returns:
            Formatted plan context string
        """
        plan = self.load_plan(path)
        return f"""=== ACTIVE PLAN ===
Task: {plan['task']}

Steps:
{self._format_steps(plan['steps'])}

Critical Files:
{self._format_files(plan['critical_files'])}
=== END PLAN ===

Continue executing from the current in_progress step. Mark steps completed as you finish them using update_plan."""

    def list_plans(self) -> list[Path]:
        """List all available plans in the plans directory.

        Returns:
            List of plan file paths, sorted by modification time (newest first)
        """
        plans = list(self.plans_dir.glob("*.md"))
        return sorted(plans, key=lambda p: p.stat().st_mtime, reverse=True)

    def get_latest_plan(self) -> Optional[Path]:
        """Get the most recently created/modified plan.

        Returns:
            Path to the latest plan, or None if no plans exist
        """
        plans = self.list_plans()
        return plans[0] if plans else None
