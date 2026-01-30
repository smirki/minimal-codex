"""PTY Shell Session Management for the Minimal Codex Agent.

Matches Codex's unified_exec module pattern:
- Persistent PTY sessions that can be reused across commands
- Session store with LRU pruning (protects 8 most recent, max 64 sessions)
- Output ring buffer with size limits
- exec_command and write_stdin operations
- Process group isolation with PDEATHSIG (Linux)
- UTF-8 boundary-safe output chunking
- Background exit watcher for async cleanup

Platform support:
- Unix: pexpect
- Windows: pywinpty
"""

import os
import sys
import time
import random
import signal
import atexit
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Callable

# For process group management on Linux
try:
    import ctypes
    HAS_CTYPES = True
except ImportError:
    HAS_CTYPES = False

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

# UTF-8 chunking constants (from async_watcher.rs)
OUTPUT_DELTA_MAX_BYTES = 8192  # Max bytes per output chunk
MAX_UTF8_SEQUENCE_LEN = 4  # Longest UTF-8 sequence is 4 bytes

# Exit watcher constants (from async_watcher.rs)
TRAILING_OUTPUT_GRACE_MS = 100  # Grace period after process exit for final output

# Linux prctl constants
PR_SET_PDEATHSIG = 1  # Set parent death signal

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


# =============================================================================
# Process Group Setup (matches Codex spawn.rs)
# =============================================================================

def create_process_group_setup(parent_pid: int) -> Callable[[], None]:
    """Create a preexec_fn for process group isolation.

    This matches Codex's spawn.rs behavior:
    1. setpgid(0, 0) - Create new process group
    2. prctl(PR_SET_PDEATHSIG, SIGTERM) - Kill on parent death (Linux only)
    3. Race condition check - If parent already died, self-terminate

    Args:
        parent_pid: PID of the parent process (captured BEFORE fork)

    Returns:
        A callable to be used as preexec_fn in pexpect.spawn()
    """
    def setup_process_group():
        """Called in child process after fork, before exec."""
        # 1. setpgid(0, 0) - Create new process group
        # This allows us to kill the entire process group if needed
        try:
            os.setpgid(0, 0)
        except OSError:
            pass  # May fail if already process group leader

        # 2. prctl(PR_SET_PDEATHSIG, SIGTERM) + race condition check - Linux only
        # Request kernel to send SIGTERM when parent dies
        if HAS_CTYPES and sys.platform.startswith("linux"):
            try:
                libc = ctypes.CDLL(None, use_errno=True)
                result = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
                if result == -1:
                    # Non-critical failure, continue
                    pass
            except Exception:
                pass  # prctl not available

            # 3. Race condition protection (Linux only, matches Codex spawn.rs)
            # If parent died between fork and prctl, the parent PID changes
            # to init (1) or a subreaper. In that case, self-terminate.
            try:
                current_ppid = os.getppid()
                if current_ppid != parent_pid:
                    # Parent already exited, self-terminate to avoid orphan
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(0.1)  # Brief delay for signal delivery
            except Exception:
                pass

    return setup_process_group


# =============================================================================
# UTF-8 Boundary-Safe Chunking (matches Codex async_watcher.rs)
# =============================================================================

def split_valid_utf8_prefix(buffer: bytearray, max_bytes: int = OUTPUT_DELTA_MAX_BYTES) -> Optional[bytes]:
    """Extract the longest valid UTF-8 prefix from buffer.

    This matches Codex's split_valid_utf8_prefix_with_max() exactly:
    1. Try to find valid UTF-8 up to max_bytes
    2. Back-scan up to 4 bytes to find valid boundary
    3. If no valid UTF-8 found, emit single byte to make progress

    Args:
        buffer: Mutable bytearray to extract from (modified in-place)
        max_bytes: Maximum bytes to extract

    Returns:
        Extracted bytes, or None if buffer is empty
    """
    if not buffer:
        return None

    max_len = min(len(buffer), max_bytes)
    split = max_len

    # Back-scan from max_len looking for valid UTF-8 boundary
    while split > 0:
        try:
            # Try to decode as UTF-8
            buffer[:split].decode("utf-8")
            # Valid UTF-8 found - extract and remove from buffer
            prefix = bytes(buffer[:split])
            del buffer[:split]
            return prefix
        except UnicodeDecodeError:
            pass

        # Prevent infinite loop: only backtrack up to 4 bytes (max UTF-8 sequence)
        if max_len - split > MAX_UTF8_SEQUENCE_LEN:
            break
        split -= 1

    # Fallback: no valid UTF-8 prefix found, emit first byte anyway
    # This ensures progress on invalid/corrupted UTF-8 streams
    byte = bytes(buffer[:1])
    del buffer[:1]
    return byte


# =============================================================================
# Exit Watcher Callbacks (for background monitoring)
# =============================================================================

# Type alias for exit callbacks
ExitCallback = Callable[[str, int, float], None]  # (process_id, exit_code, duration)

# Global registry of active sessions for cleanup on crash
_active_sessions: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_cleanup_registered = False


def _cleanup_all_sessions():
    """Cleanup handler called on interpreter exit."""
    for session in list(_active_sessions.values()):
        try:
            session.terminate()
        except Exception:
            pass


def _register_cleanup():
    """Register atexit handler for session cleanup."""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(_cleanup_all_sessions)
        _cleanup_registered = True


@dataclass
class OutputBuffer:
    """Ring buffer for PTY output with size limit.

    Matches Codex's OutputBufferState with UTF-8 boundary-safe chunking.
    """
    chunks: list = field(default_factory=list)
    total_bytes: int = 0
    max_bytes: int = OUTPUT_MAX_BYTES
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _pending: bytearray = field(default_factory=bytearray)  # For UTF-8 boundary handling

    def push_chunk(self, data: bytes):
        """Add a chunk, trimming oldest data if over limit."""
        with self._lock:
            self.chunks.append(data)
            self.total_bytes += len(data)

            # Trim from the front if over limit (matches Codex's FIFO trimming)
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

    def drain_utf8_safe(self) -> str:
        """Drain output with UTF-8 boundary-safe decoding.

        This matches Codex's process_chunk() behavior:
        - Extracts complete UTF-8 sequences
        - Preserves incomplete sequences for next drain
        - Makes progress on invalid UTF-8 by emitting single bytes
        """
        with self._lock:
            # Add all chunks to pending buffer
            self._pending.extend(b"".join(self.chunks))
            self.chunks.clear()
            self.total_bytes = 0

            # Extract UTF-8 safe chunks
            result_parts: List[bytes] = []
            while True:
                chunk = split_valid_utf8_prefix(self._pending)
                if chunk is None:
                    break
                result_parts.append(chunk)

            # Join and decode (should be valid UTF-8 now, but use replace for safety)
            return b"".join(result_parts).decode("utf-8", errors="replace")

    def snapshot(self) -> bytes:
        """Get a snapshot without draining."""
        with self._lock:
            return b"".join(self.chunks)


class PtySession:
    """A single PTY session wrapping a shell process.

    Matches Codex's UnifiedExecSession with:
    - Process group isolation (setpgid)
    - Parent death signal (PDEATHSIG on Linux)
    - Background exit watcher
    - UTF-8 safe output handling
    """

    def __init__(self, process_id: str, command: list, cwd: Path, env: dict,
                 on_exit: Optional[ExitCallback] = None):
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
        self._exit_watcher_thread = None
        self._on_exit = on_exit  # Callback when process exits

        # Synchronization for exit watcher (matches Codex's cancellation_token pattern)
        self._exit_event = threading.Event()  # Signals process has exited
        self._output_drained_event = threading.Event()  # Signals output fully collected

        # Register for cleanup on crash
        _register_cleanup()
        _active_sessions[process_id] = self

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
        """Spawn using pexpect on Unix with process group isolation.

        Matches Codex's spawn.rs behavior:
        - Creates new process group (setpgid)
        - Registers parent death signal (PDEATHSIG on Linux)
        - Protects against race condition where parent dies before signal registered
        """
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

        # Capture parent PID BEFORE fork (critical for race condition check)
        parent_pid = os.getpid()

        # Create process group setup function
        preexec_fn = create_process_group_setup(parent_pid)

        self._process = pexpect.spawn(
            cmd,
            args=args,
            cwd=str(self.cwd),
            env=spawn_env,
            encoding=None,  # Binary mode
            timeout=None,
            preexec_fn=preexec_fn,  # Process group + PDEATHSIG setup
        )

        # Start reader thread
        self._reader_thread = threading.Thread(target=self._read_output_unix, daemon=True)
        self._reader_thread.start()

        # Start exit watcher thread (matches Codex's spawn_exit_watcher)
        self._exit_watcher_thread = threading.Thread(target=self._exit_watcher, daemon=True)
        self._exit_watcher_thread.start()

        return True

    def _spawn_windows(self) -> bool:
        """Spawn using pywinpty on Windows.

        Note: Windows doesn't support setpgid or PDEATHSIG, but we still
        start the exit watcher for consistent behavior.
        """
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

        # Start exit watcher thread (matches Codex's spawn_exit_watcher)
        self._exit_watcher_thread = threading.Thread(target=self._exit_watcher, daemon=True)
        self._exit_watcher_thread.start()

        return True

    def _read_output_unix(self):
        """Background thread to read PTY output on Unix.

        Signals exit_event when process exits, allowing exit watcher
        to handle cleanup after grace period.
        """
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
            # Signal that process has exited (for exit watcher)
            self._exit_event.set()

    def _read_output_windows(self):
        """Background thread to read PTY output on Windows.

        Signals exit_event when process exits, allowing exit watcher
        to handle cleanup after grace period.
        """
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
            # Signal that process has exited (for exit watcher)
            self._exit_event.set()

    def _exit_watcher(self):
        """Background thread that monitors for process exit.

        Matches Codex's spawn_exit_watcher() from async_watcher.rs:
        1. Wait for exit_event (process exited)
        2. Wait grace period for final output (TRAILING_OUTPUT_GRACE)
        3. Signal output_drained_event
        4. Call on_exit callback if registered
        """
        # Wait for process to exit
        self._exit_event.wait()

        # Grace period for any final output to arrive
        # This matches Codex's TRAILING_OUTPUT_GRACE (100ms)
        time.sleep(TRAILING_OUTPUT_GRACE_MS / 1000.0)

        # Signal that output has been drained
        self._output_drained_event.set()

        # Calculate duration
        duration = time.time() - self.started_at
        exit_code = self._exit_code if self._exit_code is not None else -1

        # Call exit callback if registered
        if self._on_exit:
            try:
                self._on_exit(self.process_id, exit_code, duration)
            except Exception:
                pass  # Don't let callback errors crash the watcher

        # Unregister from global cleanup
        try:
            if self.process_id in _active_sessions:
                del _active_sessions[self.process_id]
        except Exception:
            pass

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

        Returns output as string with UTF-8 boundary-safe decoding.
        This matches Codex's process_chunk() behavior from async_watcher.rs.
        """
        # Wait a bit for output to accumulate
        time.sleep(min(timeout_ms, MIN_YIELD_TIME_MS) / 1000.0)

        # Use UTF-8 safe draining (preserves incomplete sequences)
        return self._output_buffer.drain_utf8_safe()

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
        """Terminate the PTY session and cleanup resources."""
        if self._process:
            try:
                if IS_WINDOWS:
                    self._process.terminate()
                else:
                    self._process.terminate(force=True)
            except Exception:
                pass
        self._has_exited = True

        # Signal exit events to unblock any waiting threads
        self._exit_event.set()
        self._output_drained_event.set()

        # Unregister from global cleanup
        try:
            if self.process_id in _active_sessions:
                del _active_sessions[self.process_id]
        except Exception:
            pass


class PtySessionManager:
    """Manages multiple PTY sessions with LRU pruning.

    Matches Codex's UnifiedExecSessionManager with:
    - Process group isolation
    - Background exit watchers
    - Automatic cleanup on process exit
    """

    def __init__(self, on_session_exit: Optional[ExitCallback] = None):
        """Initialize the session manager.

        Args:
            on_session_exit: Optional callback called when any session exits.
                             Signature: (process_id, exit_code, duration) -> None
        """
        self._sessions: Dict[str, PtySession] = OrderedDict()
        self._lock = threading.Lock()
        self._on_session_exit = on_session_exit

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

    def _make_exit_callback(self, process_id: str) -> ExitCallback:
        """Create an exit callback that removes the session and calls user callback."""
        def on_exit(pid: str, exit_code: int, duration: float):
            # Remove from session store
            with self._lock:
                if pid in self._sessions:
                    del self._sessions[pid]

            # Call user-provided callback if any
            if self._on_session_exit:
                try:
                    self._on_session_exit(pid, exit_code, duration)
                except Exception:
                    pass

        return on_exit

    def exec_command(
        self,
        command: list,
        cwd: Path,
        env: Optional[dict] = None,
        yield_time_ms: int = 2500,
    ) -> Tuple[str, str, Optional[str], Optional[int]]:
        """Execute a command in a new PTY session.

        Returns: (output, process_id or "" if exited, process_id duplicate, exit_code or None if running)

        Features matching Codex:
        - Process group isolation (setpgid)
        - Parent death signal (PDEATHSIG on Linux)
        - Background exit watcher with automatic cleanup
        """
        if not HAS_PTY:
            return self._fallback_exec(command, cwd, env)

        with self._lock:
            self._prune_if_needed()
            process_id = self._generate_process_id()

        # Create exit callback for automatic cleanup
        exit_callback = self._make_exit_callback(process_id)

        session = PtySession(
            process_id=process_id,
            command=command,
            cwd=cwd,
            env=env or {},
            on_exit=exit_callback,  # Background exit watcher will call this
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
