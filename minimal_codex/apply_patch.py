"""Apply patch implementation matching Codex CLI's exact format.

Supports:
- *** Add File: path/to/new.py
- *** Delete File: path/to/old.py
- *** Update File: path/to/file.py
- *** Move to: path/to/newname.py
- @@ context markers
- +/- line additions/removals
- Fuzzy matching (4 levels, matching seek_sequence.rs exactly)
"""

import re
from pathlib import Path
from typing import Optional


# Unicode normalization map (matching Codex's seek_sequence.rs)
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
    """Normalize Unicode characters for fuzzy matching (matching Codex's Rust impl)."""
    result = []
    for c in s.strip():
        result.append(UNICODE_NORMALIZATIONS.get(c, c))
    return ''.join(result)


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

    if len(pattern) > len(lines):
        return None

    # When eof is set, start searching from the end
    search_start = len(lines) - len(pattern) if eof else start

    # 1. Exact match first
    for i in range(search_start, len(lines) - len(pattern) + 1):
        if lines[i:i + len(pattern)] == pattern:
            return i

    # 2. rstrip match (trim trailing whitespace)
    for i in range(search_start, len(lines) - len(pattern) + 1):
        match = True
        for j, pat in enumerate(pattern):
            if lines[i + j].rstrip() != pat.rstrip():
                match = False
                break
        if match:
            return i

    # 3. trim both sides
    for i in range(search_start, len(lines) - len(pattern) + 1):
        match = True
        for j, pat in enumerate(pattern):
            if lines[i + j].strip() != pat.strip():
                match = False
                break
        if match:
            return i

    # 4. Unicode normalization (most permissive)
    for i in range(search_start, len(lines) - len(pattern) + 1):
        match = True
        for j, pat in enumerate(pattern):
            if normalize_unicode(lines[i + j]) != normalize_unicode(pat):
                match = False
                break
        if match:
            return i

    return None


def apply_patch(patch: str, cwd: Path) -> str:
    """Apply patch using Codex's exact format with fuzzy matching.

    Args:
        patch: The patch content in Codex format
        cwd: Current working directory (base for relative paths)

    Returns:
        Result message describing what was done
    """
    lines = patch.strip().split('\n')

    # Validate boundaries
    if not lines:
        return "Error: Empty patch"

    # Find the actual patch content (may have text before/after)
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "*** Begin Patch":
            start_idx = i
        if line.strip() == "*** End Patch":
            end_idx = i
            break

    if start_idx == -1:
        return "Error: Patch must contain '*** Begin Patch'"
    if end_idx == -1:
        return "Error: Patch must contain '*** End Patch'"

    # Extract just the patch content
    lines = lines[start_idx:end_idx + 1]

    results = []
    i = 1  # Skip "*** Begin Patch"

    while i < len(lines) - 1:  # Stop before "*** End Patch"
        line = lines[i]
        line_stripped = line.strip()

        if line_stripped.startswith("*** Add File: "):
            path = line_stripped[14:]
            i += 1
            content_lines = []
            while i < len(lines) - 1:
                current = lines[i]
                if current.startswith("***"):
                    break
                if current.startswith('+'):
                    content_lines.append(current[1:])  # Remove + prefix
                else:
                    # Handle lines without + prefix (might be formatting issue)
                    content_lines.append(current)
                i += 1

            full_path = cwd / path
            try:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text('\n'.join(content_lines) + '\n' if content_lines else '', encoding='utf-8')
                results.append(f"Added: {path}")
            except Exception as e:
                results.append(f"Error adding {path}: {e}")

        elif line_stripped.startswith("*** Delete File: "):
            path = line_stripped[17:]
            full_path = cwd / path
            if full_path.exists():
                try:
                    full_path.unlink()
                    results.append(f"Deleted: {path}")
                except Exception as e:
                    results.append(f"Error deleting {path}: {e}")
            else:
                results.append(f"Warning: {path} not found (already deleted?)")
            i += 1

        elif line_stripped.startswith("*** Update File: "):
            path = line_stripped[17:]
            i += 1
            result = _apply_update_hunk(lines, i, path, cwd)
            results.append(result["message"])
            i = result["next_index"]

        else:
            i += 1

    return '\n'.join(results) if results else "Patch applied (no changes)"


def _apply_update_hunk(lines: list, start_i: int, path: str, cwd: Path) -> dict:
    """Apply an update hunk to a file using Codex's seek_sequence algorithm.

    Returns:
        dict with 'message' and 'next_index'
    """
    i = start_i

    # Check for optional "*** Move to: "
    move_to = None
    if i < len(lines) - 1:
        check_line = lines[i].strip()
        if check_line.startswith("*** Move to: "):
            move_to = check_line[13:]
            i += 1

    full_path = cwd / path
    if not full_path.exists():
        # Skip to next hunk
        while i < len(lines) - 1 and not lines[i].strip().startswith("***"):
            i += 1
        return {"message": f"Error: {path} not found", "next_index": i}

    try:
        content = full_path.read_text(encoding='utf-8')
    except Exception as e:
        while i < len(lines) - 1 and not lines[i].strip().startswith("***"):
            i += 1
        return {"message": f"Error reading {path}: {e}", "next_index": i}

    # Split content into lines for line-based matching
    file_lines = content.split('\n')

    # Parse and apply all chunks for this file
    changes_made = 0
    current_pos = 0  # Track position for sequential chunks

    while i < len(lines) - 1:
        chunk_line = lines[i]

        if chunk_line.strip().startswith("***"):
            break

        # Parse @@ context marker (optional)
        change_context = None
        is_eof = False

        if chunk_line.startswith("@@"):
            if len(chunk_line.strip()) > 2:
                change_context = chunk_line[3:].strip() if chunk_line.startswith("@@ ") else None
            i += 1
            if i >= len(lines) - 1:
                break
            chunk_line = lines[i]

        # Parse change lines for this chunk
        old_lines = []
        new_lines = []

        while i < len(lines) - 1:
            l = lines[i]
            l_stripped = l.strip()

            if l_stripped.startswith("***"):
                if l_stripped == "*** End of File":
                    is_eof = True
                    i += 1
                break
            if l.startswith("@@"):
                break
            if l.startswith("-"):
                old_lines.append(l[1:])
            elif l.startswith("+"):
                new_lines.append(l[1:])
            elif l.startswith(" "):
                old_lines.append(l[1:])
                new_lines.append(l[1:])
            elif l.strip() == "":
                # Empty line - treat as context
                old_lines.append("")
                new_lines.append("")
            else:
                # Unknown format, might be end of chunk
                break
            i += 1

        # Apply the change using seek_sequence
        if old_lines:
            # Find the pattern in file_lines
            match_idx = seek_sequence(file_lines, old_lines, current_pos, is_eof)

            if match_idx is not None:
                # Replace the matched lines
                file_lines = file_lines[:match_idx] + new_lines + file_lines[match_idx + len(old_lines):]
                current_pos = match_idx + len(new_lines)
                changes_made += 1
            else:
                # Try line-by-line approach as fallback
                content_text = '\n'.join(file_lines)
                old_text = '\n'.join(old_lines)
                new_text = '\n'.join(new_lines)

                if old_text in content_text:
                    content_text = content_text.replace(old_text, new_text, 1)
                    file_lines = content_text.split('\n')
                    changes_made += 1
        elif new_lines and is_eof:
            # Pure addition at end of file
            file_lines.extend(new_lines)
            changes_made += 1

    # Reconstruct content
    content = '\n'.join(file_lines)

    # Write result
    try:
        if move_to:
            new_path = cwd / move_to
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_text(content, encoding='utf-8')
            full_path.unlink()
            message = f"Moved: {path} -> {move_to}"
        else:
            full_path.write_text(content, encoding='utf-8')
            message = f"Updated: {path} ({changes_made} changes)"
    except Exception as e:
        message = f"Error writing {path}: {e}"

    return {"message": message, "next_index": i}


def validate_patch(patch: str) -> Optional[str]:
    """Validate patch format without applying.

    Returns:
        None if valid, error message if invalid
    """
    lines = patch.strip().split('\n')

    if not lines:
        return "Empty patch"

    has_begin = any(line.strip() == "*** Begin Patch" for line in lines)
    has_end = any(line.strip() == "*** End Patch" for line in lines)

    if not has_begin:
        return "Missing '*** Begin Patch'"
    if not has_end:
        return "Missing '*** End Patch'"

    return None
