"""Apply patch implementation matching Codex CLI's exact format.

Implements the exact algorithms from Codex's Rust implementation:
- seek_sequence.rs: 4-level fuzzy matching
- parser.rs: Lenient heredoc handling
- lib.rs: compute_replacements + apply_replacements (reverse order)

Supports:
- *** Add File: path/to/new.py
- *** Delete File: path/to/old.py
- *** Update File: path/to/file.py
- *** Move to: path/to/newname.py
- @@ context markers
- +/- line additions/removals
- Lenient heredoc wrappers (<<EOF, <<'EOF', <<"EOF")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ============================================================================
# Patch Markers (exact Codex constants from parser.rs)
# ============================================================================

BEGIN_PATCH_MARKER = "*** Begin Patch"
END_PATCH_MARKER = "*** End Patch"
ADD_FILE_MARKER = "*** Add File: "
DELETE_FILE_MARKER = "*** Delete File: "
UPDATE_FILE_MARKER = "*** Update File: "
MOVE_TO_MARKER = "*** Move to: "
EOF_MARKER = "*** End of File"
CHANGE_CONTEXT_MARKER = "@@ "
EMPTY_CHANGE_CONTEXT_MARKER = "@@"


# ============================================================================
# Unicode Normalization (exact match to Codex's seek_sequence.rs)
# ============================================================================

UNICODE_NORMALIZATIONS = {
    # Various dash/hyphen code-points → ASCII '-'
    '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-',
    '\u2014': '-', '\u2015': '-', '\u2212': '-',
    # Fancy single quotes → '\''
    '\u2018': "'", '\u2019': "'", '\u201A': "'", '\u201B': "'",
    # Fancy double quotes → '"'
    '\u201C': '"', '\u201D': '"', '\u201E': '"', '\u201F': '"',
    # Non-breaking space and other odd spaces → normal space
    '\u00A0': ' ', '\u2002': ' ', '\u2003': ' ', '\u2004': ' ',
    '\u2005': ' ', '\u2006': ' ', '\u2007': ' ', '\u2008': ' ',
    '\u2009': ' ', '\u200A': ' ', '\u202F': ' ', '\u205F': ' ',
    '\u3000': ' ',
}


def normalize_unicode(s: str) -> str:
    """Normalize Unicode characters for fuzzy matching (exact Codex impl)."""
    result = []
    for c in s.strip():
        result.append(UNICODE_NORMALIZATIONS.get(c, c))
    return ''.join(result)


# ============================================================================
# seek_sequence (exact match to Codex's seek_sequence.rs)
# ============================================================================

def seek_sequence(
    lines: list[str],
    pattern: list[str],
    start: int = 0,
    eof: bool = False
) -> Optional[int]:
    """Find pattern sequence in lines with decreasing strictness.

    Matches Codex's seek_sequence.rs exactly:
    1. Exact match
    2. Trim trailing whitespace (rstrip)
    3. Trim both sides
    4. Unicode normalization

    Args:
        lines: File content lines
        pattern: Lines to search for
        start: Starting index
        eof: If True, try matching from end of file first

    Returns:
        Starting index of match, or None if not found
    """
    if not pattern:
        return start

    # Pattern longer than input - impossible match
    if len(pattern) > len(lines):
        return None

    # When eof is set, start searching from the end
    search_start = len(lines) - len(pattern) if eof else start

    # Level 1: Exact match
    for i in range(search_start, len(lines) - len(pattern) + 1):
        if lines[i:i + len(pattern)] == pattern:
            return i

    # Level 2: trim_end (rstrip) match
    for i in range(search_start, len(lines) - len(pattern) + 1):
        match = True
        for j, pat in enumerate(pattern):
            if lines[i + j].rstrip() != pat.rstrip():
                match = False
                break
        if match:
            return i

    # Level 3: trim both sides
    for i in range(search_start, len(lines) - len(pattern) + 1):
        match = True
        for j, pat in enumerate(pattern):
            if lines[i + j].strip() != pat.strip():
                match = False
                break
        if match:
            return i

    # Level 4: Unicode normalization (most permissive)
    for i in range(search_start, len(lines) - len(pattern) + 1):
        match = True
        for j, pat in enumerate(pattern):
            if normalize_unicode(lines[i + j]) != normalize_unicode(pat):
                match = False
                break
        if match:
            return i

    return None


# ============================================================================
# Patch Data Classes (from parser.rs)
# ============================================================================

@dataclass
class UpdateFileChunk:
    """A chunk within an Update File hunk."""
    change_context: Optional[str]  # Context line (class/function hint)
    old_lines: list[str]
    new_lines: list[str]
    is_end_of_file: bool = False


@dataclass
class Hunk:
    """Base class for file operation hunks."""
    pass


@dataclass
class AddFile(Hunk):
    path: Path
    contents: str


@dataclass
class DeleteFile(Hunk):
    path: Path


@dataclass
class UpdateFile(Hunk):
    path: Path
    move_path: Optional[Path]
    chunks: list[UpdateFileChunk]


# ============================================================================
# Patch Parser (exact match to Codex's parser.rs)
# ============================================================================

def parse_patch(patch: str, lenient: bool = True) -> list[Hunk]:
    """Parse patch text into hunks.

    Args:
        patch: Raw patch text
        lenient: If True, handle heredoc wrappers (<<EOF ... EOF)

    Returns:
        List of parsed hunks
    """
    lines = patch.strip().split('\n')

    # Check boundaries
    if not _check_patch_boundaries_strict(lines):
        if lenient:
            lines = _check_patch_boundaries_lenient(lines)
            if lines is None:
                raise ValueError("Invalid patch: missing Begin/End markers")
        else:
            raise ValueError("Invalid patch: missing Begin/End markers")

    # Parse hunks
    hunks = []
    i = 1  # Skip "*** Begin Patch"
    last_line = len(lines) - 1  # Before "*** End Patch"

    while i < last_line:
        hunk, consumed = _parse_one_hunk(lines, i)
        if hunk:
            hunks.append(hunk)
        i += consumed

    return hunks


def _check_patch_boundaries_strict(lines: list[str]) -> bool:
    """Check if patch has valid Begin/End markers."""
    if len(lines) < 2:
        return False
    return (lines[0].strip() == BEGIN_PATCH_MARKER and
            lines[-1].strip() == END_PATCH_MARKER)


def _check_patch_boundaries_lenient(lines: list[str]) -> Optional[list[str]]:
    """Handle heredoc wrappers: <<EOF, <<'EOF', <<"EOF" (from parser.rs)"""
    if len(lines) < 4:
        return None

    first = lines[0].strip()
    last = lines[-1].strip()

    # Check for heredoc markers
    if first in ("<<EOF", "<<'EOF'", '<<"EOF"') and last.endswith("EOF"):
        inner = lines[1:-1]
        if _check_patch_boundaries_strict(inner):
            return inner

    return None


def _parse_one_hunk(lines: list[str], start: int) -> tuple[Optional[Hunk], int]:
    """Parse a single hunk starting at line index."""
    line = lines[start].strip()

    if line.startswith(ADD_FILE_MARKER):
        path = line[len(ADD_FILE_MARKER):]
        contents = []
        i = start + 1
        while i < len(lines) and lines[i].startswith('+'):
            contents.append(lines[i][1:])
            i += 1
        return AddFile(Path(path), '\n'.join(contents) + '\n' if contents else ''), i - start

    elif line.startswith(DELETE_FILE_MARKER):
        path = line[len(DELETE_FILE_MARKER):]
        return DeleteFile(Path(path)), 1

    elif line.startswith(UPDATE_FILE_MARKER):
        path = line[len(UPDATE_FILE_MARKER):]
        i = start + 1

        # Check for Move to:
        move_path = None
        if i < len(lines) and lines[i].strip().startswith(MOVE_TO_MARKER):
            move_path = Path(lines[i].strip()[len(MOVE_TO_MARKER):])
            i += 1

        # Parse chunks
        chunks = []
        is_first_chunk = True
        while i < len(lines):
            # Skip blank lines between chunks
            if lines[i].strip() == "":
                i += 1
                continue

            # Stop at next file operation
            if lines[i].strip().startswith("***") and not lines[i].strip() == EOF_MARKER:
                break

            chunk, consumed = _parse_update_chunk(lines, i, is_first_chunk)
            if chunk:
                chunks.append(chunk)
            i += consumed
            is_first_chunk = False

        return UpdateFile(Path(path), move_path, chunks), i - start

    return None, 1


def _parse_update_chunk(
    lines: list[str],
    start: int,
    allow_missing_context: bool
) -> tuple[Optional[UpdateFileChunk], int]:
    """Parse an update chunk (context + changes)."""
    if start >= len(lines):
        return None, 0

    line = lines[start]
    change_context = None
    i = start

    # Parse @@ context marker
    if line.strip() == EMPTY_CHANGE_CONTEXT_MARKER:
        i += 1
    elif line.strip().startswith(CHANGE_CONTEXT_MARKER):
        change_context = line.strip()[len(CHANGE_CONTEXT_MARKER):]
        i += 1
    elif not allow_missing_context:
        return None, 1

    # Parse change lines
    old_lines = []
    new_lines = []
    is_eof = False

    while i < len(lines):
        l = lines[i]

        if l.strip() == EOF_MARKER:
            is_eof = True
            i += 1
            break

        if l.strip().startswith("***") or l.startswith("@@"):
            break

        if l == "" or (len(l) > 0 and l[0] not in " +-"):
            # Empty line = context
            if l == "":
                old_lines.append("")
                new_lines.append("")
                i += 1
                continue
            break

        if l.startswith(" "):
            old_lines.append(l[1:])
            new_lines.append(l[1:])
        elif l.startswith("-"):
            old_lines.append(l[1:])
        elif l.startswith("+"):
            new_lines.append(l[1:])

        i += 1

    if not old_lines and not new_lines:
        return None, i - start

    return UpdateFileChunk(change_context, old_lines, new_lines, is_eof), i - start


# ============================================================================
# compute_replacements + apply_replacements (exact match to Codex's lib.rs)
# ============================================================================

def compute_replacements(
    original_lines: list[str],
    path: Path,
    chunks: list[UpdateFileChunk]
) -> list[tuple[int, int, list[str]]]:
    """Compute replacements needed to transform file.

    Returns list of (start_index, old_len, new_lines).
    """
    replacements = []
    line_index = 0

    for chunk in chunks:
        # If chunk has context, use seek_sequence to find it
        if chunk.change_context:
            idx = seek_sequence(
                original_lines,
                [chunk.change_context],
                line_index,
                False
            )
            if idx is not None:
                line_index = idx + 1
            else:
                raise ValueError(f"Failed to find context '{chunk.change_context}' in {path}")

        if not chunk.old_lines:
            # Pure addition at end
            insertion_idx = len(original_lines)
            if original_lines and original_lines[-1] == "":
                insertion_idx -= 1
            replacements.append((insertion_idx, 0, chunk.new_lines))
            continue

        # Find old_lines in file
        pattern = chunk.old_lines
        new_slice = chunk.new_lines

        found = seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file)

        # Handle trailing empty line edge case
        if found is None and pattern and pattern[-1] == "":
            pattern = pattern[:-1]
            if new_slice and new_slice[-1] == "":
                new_slice = new_slice[:-1]
            found = seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file)

        if found is not None:
            replacements.append((found, len(pattern), list(new_slice)))
            line_index = found + len(pattern)
        else:
            raise ValueError(f"Failed to find expected lines in {path}")

    # Sort by start index
    replacements.sort(key=lambda x: x[0])
    return replacements


def apply_replacements(
    lines: list[str],
    replacements: list[tuple[int, int, list[str]]]
) -> list[str]:
    """Apply replacements in reverse order to avoid index shifting."""
    result = list(lines)

    # Apply in reverse order
    for start_idx, old_len, new_segment in reversed(replacements):
        # Remove old lines
        for _ in range(old_len):
            if start_idx < len(result):
                result.pop(start_idx)

        # Insert new lines
        for offset, new_line in enumerate(new_segment):
            result.insert(start_idx + offset, new_line)

    return result


# ============================================================================
# Main apply_patch Function
# ============================================================================

def apply_patch(patch: str, cwd: Path) -> str:
    """Apply patch using exact Codex algorithm.

    Args:
        patch: The patch content in Codex format
        cwd: Current working directory (base for relative paths)

    Returns:
        Result message describing what was done
    """
    try:
        hunks = parse_patch(patch, lenient=True)
    except ValueError as e:
        return f"Error: {e}"

    results = []

    for hunk in hunks:
        if isinstance(hunk, AddFile):
            full_path = cwd / hunk.path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(hunk.contents, encoding='utf-8')
            results.append(f"A {hunk.path}")

        elif isinstance(hunk, DeleteFile):
            full_path = cwd / hunk.path
            if full_path.exists():
                full_path.unlink()
                results.append(f"D {hunk.path}")
            else:
                results.append(f"Warning: {hunk.path} not found")

        elif isinstance(hunk, UpdateFile):
            full_path = cwd / hunk.path
            if not full_path.exists():
                results.append(f"Error: {hunk.path} not found")
                continue

            try:
                content = full_path.read_text(encoding='utf-8')
            except Exception as e:
                results.append(f"Error reading {hunk.path}: {e}")
                continue

            original_lines = content.split('\n')

            # Drop trailing empty line (matches Codex behavior)
            if original_lines and original_lines[-1] == "":
                original_lines.pop()

            try:
                replacements = compute_replacements(original_lines, hunk.path, hunk.chunks)
                new_lines = apply_replacements(original_lines, replacements)

                # Ensure trailing newline
                if not new_lines or new_lines[-1] != "":
                    new_lines.append("")

                new_content = '\n'.join(new_lines)

                if hunk.move_path:
                    dest = cwd / hunk.move_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(new_content, encoding='utf-8')
                    full_path.unlink()
                    results.append(f"M {hunk.move_path}")
                else:
                    full_path.write_text(new_content, encoding='utf-8')
                    results.append(f"U {hunk.path}")

            except ValueError as e:
                results.append(f"Error: {e}")

    return "Success. Updated the following files:\n" + '\n'.join(results) if results else "No changes"


def validate_patch(patch: str) -> Optional[str]:
    """Validate patch format without applying.

    Returns:
        None if valid, error message if invalid
    """
    try:
        parse_patch(patch, lenient=True)
        return None
    except ValueError as e:
        return str(e)
