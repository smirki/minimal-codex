"""PTY Shell Session Management for the Minimal Codex Agent.

Matches Codex's unified_exec module pattern:
- Persistent PTY sessions that can be reused across commands
- Session store with LRU pruning (protects 8 most recent, max 64 sessions)
- Output ring buffer with size limits
- exec_command and write_stdin operations

Platform support:
- Unix: pexpect
- Windows: pywinpty
"""

import os
import sys
import time
import random
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

# Platform-specific PTY support
IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    try:
        import winpty
        HAS_PTY = True
    except ImportError:
        HAS_PTY = False
else:
    try:
        import pexpect
        HAS_PTY = True
    except ImportError:
        HAS_PTY = False


# Constants matching Codex's unified_exec/mod.rs
MAX_SESSIONS = 64
WARNING_SESSIONS = 60
PROTECTED_SESSIONS = 8  # Most recent sessions protected from pruning
OUTPUT_MAX_BYTES = 1024 * 1024  # 1 MiB ring buffer
MIN_YIELD_TIME_MS = 250
MAX_YIELD_TIME_MS = 30_000
DEFAULT_MAX_OUTPUT_TOKENS = 10_000

# Environment variables for PTY sessions (matches UNIFIED_EXEC_ENV)
PTY_ENV = {
    "NO_COLOR": "1",
    "TERM": "dumb",
    "LANG": "C.UTF-8",
    "LC_CTYPE": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "COLORTERM": "",
    "PAGER": "cat",
    "GIT_PAGER": "cat",
}


@dataclass
class OutputBuffer:
    """Ring buffer for PTY output with size limit.

    Matches Codex's OutputBufferState.
    """
    chunks: list = field(default_factory=list)
    total_bytes: int = 0
    max_bytes: int = OUTPUT_MAX_BYTES
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push_chunk(self, data: bytes):
        """Add a chunk, trimming oldest data if over limit."""
        with self._lock:
            self.chunks.append(data)
            self.total_bytes += len(data)

            # Trim from the front if over limit
            while self.total_bytes > self.max_bytes and self.chunks:
                removed = self.chunks.pop(0)
                self.total_bytes -= len(removed)

    def drain(self) -> bytes:
        """Drain all accumulated output."""
        with self._lock:
            result = b"".join(self.chunks)
            self.chunks.clear()
            self.total_bytes = 0
            return result

    def snapshot(self) -> bytes:
        """Get a snapshot without draining."""
        with self._lock:
            return b"".join(self.chunks)


class PtySession:
    """A single PTY session wrapping a shell process.

    Matches Codex's UnifiedExecSession.
    """

    def __init__(self, process_id: str, command: list, cwd: Path, env: dict):
        self.process_id = process_id
        self.command = command
        self.cwd = cwd
        self.env = env
        self.started_at = time.time()
        self.last_used = time.time()
        self._output_buffer = OutputBuffer()
        self._exit_code: Optional[int] = None
        self._has_exited = False
        self._process = None
        self._reader_thread = None

    def spawn(self) -> bool:
        """Spawn the PTY process. Returns True on success."""
        if not HAS_PTY:
            return False

        try:
            if IS_WINDOWS:
                return self._spawn_windows()
            else:
                return self._spawn_unix()
        except Exception as e:
            self._has_exited = True
            self._exit_code = -1
            return False

    def _spawn_unix(self) -> bool:
        """Spawn using pexpect on Unix."""
        # Build the command
        if len(self.command) == 1:
            cmd = self.command[0]
            args = []
        else:
            cmd = self.command[0]
            args = self.command[1:]

        # Merge environment
        spawn_env = os.environ.copy()
        spawn_env.update(self.env)
        spawn_env.update(PTY_ENV)

        self._process = pexpect.spawn(
            cmd,
            args=args,
            cwd=str(self.cwd),
            env=spawn_env,
            encoding=None,  # Binary mode
            timeout=None,
        )

        # Start reader thread
        self._reader_thread = threading.Thread(target=self._read_output_unix, daemon=True)
        self._reader_thread.start()
        return True

    def _spawn_windows(self) -> bool:
        """Spawn using pywinpty on Windows."""
        # Build command line
        cmdline = " ".join(self.command)

        # Merge environment
        spawn_env = os.environ.copy()
        spawn_env.update(self.env)
        spawn_env.update(PTY_ENV)

        self._process = winpty.PtyProcess.spawn(
            cmdline,
            cwd=str(self.cwd),
            env=spawn_env,
        )

        # Start reader thread
        self._reader_thread = threading.Thread(target=self._read_output_windows, daemon=True)
        self._reader_thread.start()
        return True

    def _read_output_unix(self):
        """Background thread to read PTY output on Unix."""
        try:
            while self._process and self._process.isalive():
                try:
                    data = self._process.read_nonblocking(size=8192, timeout=0.1)
                    if data:
                        self._output_buffer.push_chunk(data)
                except pexpect.TIMEOUT:
                    continue
                except pexpect.EOF:
                    break
        except Exception:
            pass
        finally:
            self._has_exited = True
            if self._process:
                self._exit_code = self._process.exitstatus

    def _read_output_windows(self):
        """Background thread to read PTY output on Windows."""
        try:
            while self._process and self._process.isalive():
                try:
                    data = self._process.read(8192)
                    if data:
                        if isinstance(data, str):
                            data = data.encode("utf-8", errors="replace")
                        self._output_buffer.push_chunk(data)
                except Exception:
                    time.sleep(0.1)
        except Exception:
            pass
        finally:
            self._has_exited = True
            if self._process:
                self._exit_code = self._process.exitstatus

    def write(self, data: str) -> bool:
        """Write input to the PTY."""
        if not self._process or self._has_exited:
            return False

        try:
            if IS_WINDOWS:
                self._process.write(data)
            else:
                self._process.send(data.encode("utf-8"))
            return True
        except Exception:
            return False

    def read_output(self, timeout_ms: int = 1000) -> str:
        """Read accumulated output with timeout.

        Returns output as string (UTF-8 decoded).
        """
        # Wait a bit for output to accumulate
        time.sleep(min(timeout_ms, MIN_YIELD_TIME_MS) / 1000.0)

        output = self._output_buffer.drain()
        return output.decode("utf-8", errors="replace")

    def has_exited(self) -> bool:
        """Check if the process has exited."""
        if self._has_exited:
            return True

        if self._process:
            if IS_WINDOWS:
                alive = self._process.isalive()
            else:
                alive = self._process.isalive()

            if not alive:
                self._has_exited = True
                if IS_WINDOWS:
                    self._exit_code = self._process.exitstatus
                else:
                    self._exit_code = self._process.exitstatus
        return self._has_exited

    def exit_code(self) -> Optional[int]:
        """Get the exit code if process has exited."""
        self.has_exited()  # Update state
        return self._exit_code

    def terminate(self):
        """Terminate the PTY session."""
        if self._process:
            try:
                if IS_WINDOWS:
                    self._process.terminate()
                else:
                    self._process.terminate(force=True)
            except Exception:
                pass
        self._has_exited = True


class PtySessionManager:
    """Manages multiple PTY sessions with LRU pruning.

    Matches Codex's UnifiedExecSessionManager.
    """

    def __init__(self):
        self._sessions: Dict[str, PtySession] = OrderedDict()
        self._lock = threading.Lock()

    def _generate_process_id(self) -> str:
        """Generate a unique process ID."""
        while True:
            pid = str(random.randint(1000, 99999))
            if pid not in self._sessions:
                return pid

    def _prune_if_needed(self):
        """Prune sessions if over limit. Protect 8 most recent."""
        if len(self._sessions) < MAX_SESSIONS:
            return

        # Get sessions sorted by last_used (oldest first)
        sorted_sessions = sorted(
            self._sessions.items(),
            key=lambda x: x[1].last_used
        )

        # Get the 8 most recently used (protected)
        protected_ids = {
            s[0] for s in sorted(sorted_sessions, key=lambda x: x[1].last_used, reverse=True)[:PROTECTED_SESSIONS]
        }

        # Find session to prune: prefer exited sessions outside protected set
        for pid, session in sorted_sessions:
            if pid not in protected_ids and session.has_exited():
                session.terminate()
                del self._sessions[pid]
                return

        # Fall back to LRU outside protected set
        for pid, session in sorted_sessions:
            if pid not in protected_ids:
                session.terminate()
                del self._sessions[pid]
                return

    def exec_command(
        self,
        command: list,
        cwd: Path,
        env: Optional[dict] = None,
        yield_time_ms: int = 2500,
    ) -> Tuple[str, str, Optional[str], Optional[int]]:
        """Execute a command in a new PTY session.

        Returns: (output, process_id or None if exited, exit_code or None if still running)
        """
        if not HAS_PTY:
            return self._fallback_exec(command, cwd, env)

        with self._lock:
            self._prune_if_needed()
            process_id = self._generate_process_id()

        session = PtySession(
            process_id=process_id,
            command=command,
            cwd=cwd,
            env=env or {},
        )

        if not session.spawn():
            return f"Error: Failed to spawn PTY process", "", None, -1

        # Wait for output
        yield_time_ms = max(MIN_YIELD_TIME_MS, min(yield_time_ms, MAX_YIELD_TIME_MS))
        time.sleep(yield_time_ms / 1000.0)

        output = session.read_output(yield_time_ms)
        exit_code = session.exit_code()

        if session.has_exited():
            # Short-lived command, don't persist
            return output, "", None, exit_code
        else:
            # Long-running, persist the session
            with self._lock:
                self._sessions[process_id] = session
                if len(self._sessions) >= WARNING_SESSIONS:
                    output += f"\n[Warning: {len(self._sessions)}/{MAX_SESSIONS} sessions open]"
            return output, process_id, process_id, exit_code

    def write_stdin(
        self,
        process_id: str,
        input_data: str,
        yield_time_ms: int = 2500,
    ) -> Tuple[str, Optional[str], Optional[int]]:
        """Write input to an existing session.

        Returns: (output, process_id if still alive, exit_code if exited)
        """
        with self._lock:
            session = self._sessions.get(process_id)
            if not session:
                return f"Error: Unknown session ID: {process_id}", None, None

            session.last_used = time.time()

        # Send input
        if input_data:
            session.write(input_data)
            time.sleep(0.1)  # Brief delay for process to react

        # Read output
        yield_time_ms = max(MIN_YIELD_TIME_MS, min(yield_time_ms, MAX_YIELD_TIME_MS))
        output = session.read_output(yield_time_ms)

        if session.has_exited():
            # Session ended, remove from store
            with self._lock:
                if process_id in self._sessions:
                    del self._sessions[process_id]
            return output, None, session.exit_code()
        else:
            return output, process_id, None

    def _fallback_exec(
        self,
        command: list,
        cwd: Path,
        env: Optional[dict],
    ) -> Tuple[str, str, Optional[str], Optional[int]]:
        """Fallback to subprocess when PTY not available."""
        import subprocess

        try:
            spawn_env = os.environ.copy()
            if env:
                spawn_env.update(env)
            spawn_env.update(PTY_ENV)

            result = subprocess.run(
                command,
                cwd=str(cwd),
                env=spawn_env,
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = result.stdout
            if result.stderr:
                output += f"\nstderr: {result.stderr}"

            return output, "", None, result.returncode

        except subprocess.TimeoutExpired:
            return "Command timed out", "", None, -1
        except Exception as e:
            return f"Error: {e}", "", None, -1

    def terminate_session(self, process_id: str) -> bool:
        """Terminate a specific session."""
        with self._lock:
            session = self._sessions.pop(process_id, None)
            if session:
                session.terminate()
                return True
            return False

    def terminate_all(self):
        """Terminate all sessions."""
        with self._lock:
            for session in self._sessions.values():
                session.terminate()
            self._sessions.clear()

    def list_sessions(self) -> list:
        """List all active session IDs."""
        with self._lock:
            return list(self._sessions.keys())

    def session_count(self) -> int:
        """Get the number of active sessions."""
        with self._lock:
            return len(self._sessions)
