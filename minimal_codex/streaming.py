"""Token streaming controller for the Minimal Codex Agent.

Matches Codex's streaming/controller.rs and markdown_stream.rs pattern:
- Newline-gated commit: only output complete lines as they arrive
- Buffer partial content until newline
- Finalize flushes any remaining partial content

This is a simplified Python version focused on stdout output rather than TUI.
"""

import sys
from typing import Callable, Optional


class DeltaCollector:
    """Newline-gated accumulator that commits only complete lines.

    Matches Codex's MarkdownStreamCollector logic.
    """

    def __init__(self):
        self.buffer: str = ""
        self.committed_offset: int = 0

    def clear(self):
        """Reset the collector state."""
        self.buffer = ""
        self.committed_offset = 0

    def push_delta(self, delta: str):
        """Add a delta to the buffer."""
        self.buffer += delta

    def commit_complete_lines(self) -> str:
        """Return newly completed lines (up to last newline).

        Only content up to and including the last newline is considered
        complete. Returns empty string if no new complete lines.
        """
        # Find the last newline in the buffer
        last_newline = self.buffer.rfind('\n')
        if last_newline == -1:
            return ""

        # Extract content from committed_offset to last_newline (inclusive)
        complete_end = last_newline + 1
        if complete_end <= self.committed_offset:
            return ""

        new_content = self.buffer[self.committed_offset:complete_end]
        self.committed_offset = complete_end
        return new_content

    def finalize_and_drain(self) -> str:
        """Finalize: return all remaining content including partial lines."""
        remaining = self.buffer[self.committed_offset:]
        self.clear()
        return remaining


class StreamController:
    """Controller for streaming tokens to stdout.

    Matches Codex's StreamController pattern with newline-gated output.
    """

    def __init__(self, output_fn: Optional[Callable[[str], None]] = None):
        """Initialize the stream controller.

        Args:
            output_fn: Function to call with output. Defaults to sys.stdout.write
        """
        self.collector = DeltaCollector()
        self.output_fn = output_fn or self._default_output
        self.has_seen_delta = False
        self._needs_flush = False

    def _default_output(self, text: str):
        """Default output: write to stdout and flush."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def push(self, delta: str) -> bool:
        """Push a delta; output complete lines if newline present.

        Returns True if output was emitted.
        """
        if not delta:
            return False

        self.has_seen_delta = True
        self.collector.push_delta(delta)

        if '\n' in delta:
            complete = self.collector.commit_complete_lines()
            if complete:
                self.output_fn(complete)
                return True

        return False

    def finalize(self) -> str:
        """Finalize the stream, outputting any remaining content.

        Returns the final content that was output.
        """
        remaining = self.collector.finalize_and_drain()
        if remaining:
            # Ensure final output ends with newline for clean formatting
            if not remaining.endswith('\n'):
                remaining += '\n'
            self.output_fn(remaining)

        self.has_seen_delta = False
        return remaining

    def is_idle(self) -> bool:
        """Check if there's no pending content."""
        return self.collector.committed_offset >= len(self.collector.buffer)


def stream_response(response_iterator, output_fn: Optional[Callable[[str], None]] = None) -> str:
    """Stream a response iterator, outputting tokens as they arrive.

    This is the main entry point for streaming LiteLLM responses.

    Args:
        response_iterator: An iterator yielding response chunks (from LiteLLM stream=True)
        output_fn: Optional output function, defaults to stdout

    Returns:
        The complete accumulated response text
    """
    controller = StreamController(output_fn)
    full_content = ""

    for chunk in response_iterator:
        # LiteLLM streaming chunk format
        if hasattr(chunk, 'choices') and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                content = delta.content
                full_content += content
                controller.push(content)

    # Finalize to flush any remaining partial content
    controller.finalize()

    return full_content
