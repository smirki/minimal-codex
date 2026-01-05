"""Feature flags for the Minimal Codex Agent.

Matches Codex's features.rs pattern for enabling/disabling features.
All features are ENABLED by default - use --no-* flags to disable.
"""

import os
from dataclasses import dataclass
from enum import Enum


class Feature(Enum):
    """Available feature flags."""
    PTY_SHELL = "pty_shell"           # PTY-backed persistent shell sessions
    STREAMING = "streaming"            # Token streaming to stdout
    WEB_SEARCH = "web_search"          # Web search tool
    PLAN_MODE = "plan_mode"            # Autonomous planning workflow
    SUBAGENTS = "subagents"            # Subagent invocation system


@dataclass
class FeatureSpec:
    """Specification for a feature."""
    key: str
    default_enabled: bool


# Feature specifications - all enabled by default
FEATURES = {
    Feature.PTY_SHELL: FeatureSpec("pty_shell", default_enabled=True),
    Feature.STREAMING: FeatureSpec("streaming", default_enabled=True),
    Feature.WEB_SEARCH: FeatureSpec("web_search", default_enabled=True),
    Feature.PLAN_MODE: FeatureSpec("plan_mode", default_enabled=True),
    Feature.SUBAGENTS: FeatureSpec("subagents", default_enabled=True),
}


class Features:
    """Feature flag container (matches Codex's Features struct).

    All features are enabled by default. Use disable() or environment
    variables (CODEX_FEATURE_*=0) to turn them off.
    """

    def __init__(self):
        """Initialize with all default-enabled features."""
        self._enabled = {f for f, spec in FEATURES.items() if spec.default_enabled}

    def enabled(self, feature: Feature) -> bool:
        """Check if a feature is enabled."""
        return feature in self._enabled

    def enable(self, feature: Feature) -> "Features":
        """Enable a feature. Returns self for chaining."""
        self._enabled.add(feature)
        return self

    def disable(self, feature: Feature) -> "Features":
        """Disable a feature. Returns self for chaining."""
        self._enabled.discard(feature)
        return self

    @classmethod
    def from_env(cls) -> "Features":
        """Load feature flags from environment variables.

        Environment variables:
        - CODEX_FEATURE_PTY_SHELL=0  -> disable PTY shell
        - CODEX_FEATURE_STREAMING=0  -> disable streaming
        - CODEX_FEATURE_WEB_SEARCH=0 -> disable web search
        - CODEX_FEATURE_PLAN_MODE=0  -> disable autonomous planning
        - CODEX_FEATURE_SUBAGENTS=0  -> disable subagent invocation
        """
        f = cls()

        # Check for explicit disabling via environment
        if os.environ.get("CODEX_FEATURE_PTY_SHELL") == "0":
            f.disable(Feature.PTY_SHELL)
        if os.environ.get("CODEX_FEATURE_STREAMING") == "0":
            f.disable(Feature.STREAMING)
        if os.environ.get("CODEX_FEATURE_WEB_SEARCH") == "0":
            f.disable(Feature.WEB_SEARCH)
        if os.environ.get("CODEX_FEATURE_PLAN_MODE") == "0":
            f.disable(Feature.PLAN_MODE)
        if os.environ.get("CODEX_FEATURE_SUBAGENTS") == "0":
            f.disable(Feature.SUBAGENTS)

        return f

    def __repr__(self) -> str:
        enabled_names = [f.value for f in self._enabled]
        return f"Features(enabled={enabled_names})"
