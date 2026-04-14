"""Shared tool factories for agents.

Each factory returns a list of ``ToolDefinition`` objects with handlers
bound to a specific context (repo path, Jira adapter, GitHub token, etc.).

Security: all filesystem tools validate paths stay within the repo root.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

from stary.inference.base import ToolDefinition, ToolParam

# Directories to skip when scanning repos
_IGNORED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs",
}

# Max characters returned by a single Jira tool call.  Prevents
# large epics or search results from flooding the LLM context window.
_JIRA_TOOL_MAX_CHARS = 8_000


def _truncate(text: str, limit: int = _JIRA_TOOL_MAX_CHARS) -> str:
    """Truncate *text* to *limit* characters with a notice."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated — output exceeded limit)"


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def _safe_resolve(repo_path: str, relative: str) -> Path:
    """Resolve *relative* inside *repo_path*, preventing path traversal."""
    base = Path(repo_path).resolve()
    full = (base / relative).resolve()
    if not str(full).startswith(str(base)):
        raise ValueError(f"Path '{relative}' escapes repository root")
    return full


# ---------------------------------------------------------------------------
# Filesystem tools (for Planner + Implementer)
# ---------------------------------------------------------------------------

def make_read_tools(repo_path: str) -> list[ToolDefinition]:
    """Read-only filesystem tools bound to *repo_path*."""

    def read_file(path: str) -> str:
        full = _safe_resolve(repo_path, path)
        if not full.is_file():
            return f"Error: '{path}' is not a file or does not exist."
        try:
            return full.read_text(errors="replace")
        except OSError as exc:
            return f"Error reading '{path}': {exc}"

    def list_directory(path: str = ".") -> str:
        full = _safe_resolve(repo_path, path)
        if not full.is_dir():
            return f"Error: '{path}' is not a directory."
        entries: list[str] = []
        try:
            for item in sorted(full.iterdir()):
                if item.name in _IGNORED_DIRS or item.name.endswith(".egg-info"):
                    continue
                suffix = "/" if item.is_dir() else ""
                entries.append(f"{item.name}{suffix}")
        except OSError as exc:
            return f"Error listing '{path}': {exc}"
        return "\n".join(entries) if entries else "(empty directory)"

    def search_code(pattern: str, path: str = ".") -> str:
        full = _safe_resolve(repo_path, path)
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "--include=*.ts",
                 "--include=*.js", "--include=*.yaml", "--include=*.yml",
                 "--include=*.json", "--include=*.toml", "--include=*.md",
                 "-l", pattern, str(full)],
                capture_output=True, text=True, timeout=30,
            )
            if result.stdout.strip():
                # Make paths relative to repo root
                base = Path(repo_path).resolve()
                lines: list[str] = []
                for line in result.stdout.strip().splitlines()[:50]:
                    try:
                        rel = str(Path(line).resolve().relative_to(base))
                        lines.append(rel)
                    except ValueError:
                        lines.append(line)
                return "\n".join(lines)
            return "No matches found."
        except (subprocess.TimeoutExpired, OSError) as exc:
            return f"Search error: {exc}"

    def search_code_content(pattern: str, path: str = ".") -> str:
        full = _safe_resolve(repo_path, path)
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "--include=*.ts",
                 "--include=*.js", "--include=*.yaml", "--include=*.yml",
                 "--include=*.json", "--include=*.toml", "--include=*.md",
                 pattern, str(full)],
                capture_output=True, text=True, timeout=30,
            )
            if result.stdout.strip():
                base = Path(repo_path).resolve()
                lines: list[str] = []
                for line in result.stdout.strip().splitlines()[:80]:
                    try:
                        colon = line.index(":")
                        fpath = str(Path(line[:colon]).resolve().relative_to(base))
                        lines.append(f"{fpath}:{line[colon+1:]}")
                    except (ValueError, IndexError):
                        lines.append(line)
                return "\n".join(lines)
            return "No matches found."
        except (subprocess.TimeoutExpired, OSError) as exc:
            return f"Search error: {exc}"

    return [
        ToolDefinition(
            name="read_file",
            description="Read the full contents of a file in the repository.",
            parameters=[
                ToolParam("path", "string", "Repo-relative path to the file"),
            ],
            handler=read_file,
        ),
        ToolDefinition(
            name="list_directory",
            description="List files and subdirectories in a directory.",
            parameters=[
                ToolParam("path", "string", "Repo-relative directory path (use '.' for root)", required=False),
            ],
            handler=list_directory,
        ),
        ToolDefinition(
            name="search_files",
            description="Search for files whose content matches a pattern. Returns matching file paths.",
            parameters=[
                ToolParam("pattern", "string", "Text or regex pattern to search for"),
                ToolParam("path", "string", "Directory to search in (default: repo root)", required=False),
            ],
            handler=search_code,
        ),
        ToolDefinition(
            name="search_code",
            description="Search for a pattern in file contents. Returns matching lines with file paths and line numbers.",
            parameters=[
                ToolParam("pattern", "string", "Text or regex pattern to search for"),
                ToolParam("path", "string", "Directory to search in (default: repo root)", required=False),
            ],
            handler=search_code_content,
        ),
    ]


def make_write_tools(repo_path: str) -> list[ToolDefinition]:
    """Write tools for the Implementer, bound to *repo_path*."""

    def write_file(path: str, content: str) -> str:
        full = _safe_resolve(repo_path, path)
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)
            return f"Successfully wrote {len(content)} chars to '{path}'."
        except OSError as exc:
            return f"Error writing '{path}': {exc}"

    def modify_file(path: str, old_text: str, new_text: str) -> str:
        full = _safe_resolve(repo_path, path)
        if not full.is_file():
            return f"Error: '{path}' does not exist."
        try:
            text = full.read_text(errors="replace")
        except OSError as exc:
            return f"Error reading '{path}': {exc}"

        if old_text not in text:
            # Try whitespace-normalised match
            norm_old = " ".join(old_text.split())
            norm_text = " ".join(text.split())
            if norm_old not in norm_text:
                return (
                    f"Error: search text not found in '{path}'. "
                    "Ensure the old_text exactly matches the current file content "
                    "(including whitespace and indentation). "
                    "Use read_file to check the current content first."
                )
            # Whitespace-normalised match: find original span
            idx = norm_text.index(norm_old)
            # Map normalised position back to original
            orig_pos = 0
            norm_pos = 0
            while norm_pos < idx and orig_pos < len(text):
                if text[orig_pos].isspace():
                    while orig_pos < len(text) and text[orig_pos].isspace():
                        orig_pos += 1
                    norm_pos += 1
                else:
                    orig_pos += 1
                    norm_pos += 1
            start = orig_pos
            matched = 0
            while matched < len(norm_old) and orig_pos < len(text):
                if text[orig_pos].isspace():
                    while orig_pos < len(text) and text[orig_pos].isspace():
                        orig_pos += 1
                    matched += 1
                else:
                    orig_pos += 1
                    matched += 1
            text = text[:start] + new_text + text[orig_pos:]
        else:
            text = text.replace(old_text, new_text, 1)

        try:
            full.write_text(text)
            return f"Successfully modified '{path}'."
        except OSError as exc:
            return f"Error writing '{path}': {exc}"

    def delete_file(path: str) -> str:
        full = _safe_resolve(repo_path, path)
        if not full.is_file():
            return f"Error: '{path}' does not exist."
        try:
            full.unlink()
            return f"Deleted '{path}'."
        except OSError as exc:
            return f"Error deleting '{path}': {exc}"

    return [
        ToolDefinition(
            name="write_file",
            description=(
                "Create a new file or overwrite an existing file with the "
                "given content. Parent directories are created automatically."
            ),
            parameters=[
                ToolParam("path", "string", "Repo-relative path for the file"),
                ToolParam("content", "string", "Full content to write"),
            ],
            handler=write_file,
        ),
        ToolDefinition(
            name="modify_file",
            description=(
                "Apply a search-and-replace edit to an existing file. "
                "old_text must exactly match a section of the file. "
                "Include 2-3 unchanged context lines around the target. "
                "Use read_file first to see current content."
            ),
            parameters=[
                ToolParam("path", "string", "Repo-relative path to the file"),
                ToolParam("old_text", "string", "Exact text to find in the file"),
                ToolParam("new_text", "string", "Text to replace it with"),
            ],
            handler=modify_file,
        ),
        ToolDefinition(
            name="delete_file",
            description="Delete a file from the repository.",
            parameters=[
                ToolParam("path", "string", "Repo-relative path to delete"),
            ],
            handler=delete_file,
        ),
    ]


# ---------------------------------------------------------------------------
# Shell tools (for Implementer)
# ---------------------------------------------------------------------------

_COMMAND_ALLOWLIST = {
    "git", "find", "ls", "wc", "grep", "cat", "head", "tail",
    "sort", "uniq", "xargs", "echo", "test", "mkdir", "rm",
    "cp", "mv", "touch", "dirname", "basename", "sed", "awk",
}


def make_shell_tools(repo_path: str) -> list[ToolDefinition]:
    """Shell command tool for the Implementer, scoped to *repo_path*."""

    def run_command(command: str) -> str:
        # Validate the base command is in the allowlist
        parts = command.strip().split()
        if not parts:
            return "Error: empty command."
        base_cmd = os.path.basename(parts[0])
        if base_cmd not in _COMMAND_ALLOWLIST:
            return (
                f"Error: command '{base_cmd}' is not allowed. "
                f"Allowed commands: {', '.join(sorted(_COMMAND_ALLOWLIST))}"
            )
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output_parts: list[str] = []
            if result.stdout:
                stdout = result.stdout
                if len(stdout) > 50_000:
                    stdout = stdout[:50_000] + "\n... (truncated)"
                output_parts.append(stdout)
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr[:10_000]}")
            if result.returncode != 0:
                output_parts.append(f"Exit code: {result.returncode}")
            return "\n".join(output_parts) if output_parts else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 120s."
        except OSError as exc:
            return f"Error running command: {exc}"

    return [
        ToolDefinition(
            name="run_command",
            description=(
                "Run a shell command in the repository root directory. "
                "Allowed commands: git, find, ls, grep, cat, head, tail, "
                "sort, uniq, xargs, echo, test, mkdir, rm, cp, mv, touch, "
                "dirname, basename, sed, awk, wc. "
                "Use this for bulk operations like 'git rm' to untrack many "
                "files, 'find' to locate files, or piped commands."
            ),
            parameters=[
                ToolParam("command", "string", "Shell command to execute"),
            ],
            handler=run_command,
        ),
    ]


# ---------------------------------------------------------------------------
# Jira tools (for TaskReader)
# ---------------------------------------------------------------------------

def make_jira_tools(jira_adapter) -> list[ToolDefinition]:
    """Tools for fetching Jira ticket information."""

    # Discover the instance-specific epic-link custom field once.
    _epic_field = getattr(jira_adapter, "get_epic_link_field", lambda: "")()

    def fetch_ticket(issue_key: str) -> str:
        try:
            request_fields = [
                "summary", "description", "status", "issuetype",
                "priority", "labels", "components", "fixVersions",
                "issuelinks", "subtasks", "parent",
            ]
            if _epic_field:
                request_fields.append(_epic_field)

            issue = jira_adapter.get_issue(issue_key, fields=request_fields)
            f = issue.fields or {}

            status = (f.get("status") or {}).get("name", "")
            issuetype = (f.get("issuetype") or {}).get("name", "")
            priority = (f.get("priority") or {}).get("name", "")
            labels = f.get("labels") or []
            components = [c.get("name", "") for c in (f.get("components") or [])]
            fix_versions = [v.get("name", "") for v in (f.get("fixVersions") or [])]

            # Epic / parent detection
            parent = f.get("parent") or {}
            parent_key = parent.get("key", "")
            epic_link = (f.get(_epic_field) or "") if _epic_field else ""
            epic_display = epic_link or parent_key or ""

            # Linked issues summary
            links_raw = f.get("issuelinks") or []
            link_lines: list[str] = []
            for link in links_raw:
                lt = (link.get("type") or {}).get("name", "")
                for direction in ("inwardIssue", "outwardIssue"):
                    target = link.get(direction)
                    if target:
                        t_fields = target.get("fields") or {}
                        t_status = (t_fields.get("status") or {}).get("name", "")
                        link_lines.append(
                            f"  - [{lt}] {target.get('key', '')} "
                            f"({t_status}): {t_fields.get('summary', '')}"
                        )

            # Subtasks summary
            subtasks_raw = f.get("subtasks") or []
            subtask_lines: list[str] = []
            for st in subtasks_raw:
                st_f = st.get("fields") or {}
                st_status = (st_f.get("status") or {}).get("name", "")
                subtask_lines.append(
                    f"  - {st.get('key', '')} ({st_status}): "
                    f"{st_f.get('summary', '')}"
                )

            parts = [
                f"Key: {issue.key}",
                f"Type: {issuetype}",
                f"Status: {status}",
                f"Priority: {priority}",
                f"Summary: {issue.summary}",
            ]
            if epic_display:
                parts.append(f"Epic/Parent: {epic_display}")
            if labels:
                parts.append(f"Labels: {', '.join(labels)}")
            if components:
                parts.append(f"Components: {', '.join(components)}")
            if fix_versions:
                parts.append(f"Fix Versions: {', '.join(fix_versions)}")
            if link_lines:
                parts.append("Linked Issues:\n" + "\n".join(link_lines))
            if subtask_lines:
                parts.append("Subtasks:\n" + "\n".join(subtask_lines))
            parts.append(f"Description:\n{issue.description}")

            return _truncate("\n".join(parts))
        except Exception as exc:
            logger.error("fetch_ticket(%s) failed: %s", issue_key, exc)
            return f"Error fetching ticket {issue_key}: {exc}"

    def get_comments(issue_key: str) -> str:
        try:
            comments = jira_adapter.get_comments(issue_key)
            if not comments:
                return "No comments on this ticket."
            parts: list[str] = []
            for c in comments[:20]:
                author = getattr(c, "author", "") or "Unknown"
                body = getattr(c, "body", "") or ""
                parts.append(f"[{author}]: {body[:500]}")
            return _truncate("\n---\n".join(parts))
        except Exception as exc:
            return f"Error fetching comments for {issue_key}: {exc}"

    def get_linked_issues(issue_key: str) -> str:
        try:
            links = jira_adapter.get_linked_issues(issue_key)
            if not links:
                return "No linked issues."
            parts: list[str] = []
            for lnk in links:
                parts.append(
                    f"[{lnk['link_type']} / {lnk['direction']}] "
                    f"{lnk['key']} ({lnk['status']}): {lnk['summary']}"
                )
            return _truncate("\n".join(parts))
        except Exception as exc:
            return f"Error fetching linked issues for {issue_key}: {exc}"

    def get_subtasks(issue_key: str) -> str:
        try:
            subtasks = jira_adapter.get_subtasks(issue_key)
            if not subtasks:
                return "No subtasks."
            parts: list[str] = []
            for st in subtasks:
                parts.append(f"{st['key']} ({st['status']}): {st['summary']}")
            return _truncate("\n".join(parts))
        except Exception as exc:
            return f"Error fetching subtasks for {issue_key}: {exc}"

    def get_epic_children(epic_key: str) -> str:
        try:
            children = jira_adapter.get_epic_children(epic_key)
            if not children:
                return "No children found for this epic."
            parts: list[str] = []
            for ch in children:
                ch_fields = ch.fields or {}
                status = (ch_fields.get("status") or {}).get("name", "")
                itype = (ch_fields.get("issuetype") or {}).get("name", "")
                assignee = (ch_fields.get("assignee") or {}).get("displayName", "Unassigned")
                parts.append(
                    f"{ch.key} [{itype}] ({status}, {assignee}): {ch.summary}"
                )
            return _truncate("\n".join(parts))
        except Exception as exc:
            return f"Error fetching epic children for {epic_key}: {exc}"

    def find_similar_resolved(issue_key: str) -> str:
        try:
            results = jira_adapter.find_similar_resolved(issue_key)
            if not results:
                return "No similar resolved tickets found."
            parts: list[str] = []
            for r in results:
                r_fields = r.fields or {}
                resolution = (r_fields.get("resolution") or {}).get("name", "")
                parts.append(f"{r.key} (resolved: {resolution}): {r.summary}")
            return _truncate("\n".join(parts))
        except Exception as exc:
            return f"Error finding similar resolved tickets: {exc}"

    def search_issues(jql: str) -> str:
        try:
            results = jira_adapter.search_issues(jql, max_results=10)
            if not results:
                return "No matching issues found."
            parts: list[str] = []
            for issue in results:
                parts.append(f"{issue.key}: {issue.summary}")
            return _truncate("\n".join(parts))
        except Exception as exc:
            return f"Error searching Jira: {exc}"

    return [
        ToolDefinition(
            name="fetch_ticket",
            description=(
                "Fetch a Jira ticket by its issue key (e.g. PROJ-123). "
                "Returns summary, description, status, priority, labels, "
                "components, linked issues, subtasks, and epic/parent info."
            ),
            parameters=[
                ToolParam("issue_key", "string", "Jira issue key, e.g. PROJ-123"),
            ],
            handler=fetch_ticket,
        ),
        ToolDefinition(
            name="get_comments",
            description="Get comments on a Jira issue.",
            parameters=[
                ToolParam("issue_key", "string", "Jira issue key"),
            ],
            handler=get_comments,
        ),
        ToolDefinition(
            name="get_linked_issues",
            description=(
                "Get all issues linked to a ticket (blocks, is-blocked-by, "
                "relates-to, duplicates, etc.) with their status and summary."
            ),
            parameters=[
                ToolParam("issue_key", "string", "Jira issue key"),
            ],
            handler=get_linked_issues,
        ),
        ToolDefinition(
            name="get_subtasks",
            description="Get all subtasks of a Jira issue with their status.",
            parameters=[
                ToolParam("issue_key", "string", "Jira issue key"),
            ],
            handler=get_subtasks,
        ),
        ToolDefinition(
            name="get_epic_children",
            description=(
                "Get all stories/tasks that belong to an epic. Returns each "
                "child's key, type, status, assignee, and summary. Use this "
                "to understand the scope of the epic the current ticket "
                "belongs to."
            ),
            parameters=[
                ToolParam("epic_key", "string", "Epic issue key, e.g. PROJ-100"),
            ],
            handler=get_epic_children,
        ),
        ToolDefinition(
            name="find_similar_resolved",
            description=(
                "Find recently resolved tickets in the same project and "
                "components as the given ticket. Useful for finding "
                "implementation precedent and patterns from past work."
            ),
            parameters=[
                ToolParam("issue_key", "string", "Jira issue key to find similar resolved tickets for"),
            ],
            handler=find_similar_resolved,
        ),
        ToolDefinition(
            name="search_issues",
            description=(
                "Search Jira issues with a raw JQL query. Use this as a "
                "fallback when the other Jira tools don't cover your needs."
            ),
            parameters=[
                ToolParam("jql", "string", "JQL query string"),
            ],
            handler=search_issues,
        ),
    ]


# ---------------------------------------------------------------------------
# GitHub file URL parsing
# ---------------------------------------------------------------------------

_GITHUB_FILE_URL_RE = re.compile(
    r"https?://github\.com/"
    r"(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"/(?P<kind>blob|tree)/(?P<rest>.+)",
)

_LINE_RANGE_RE = re.compile(r"#L(?P<start>\d+)(?:-L?(?P<end>\d+))?$")


def _parse_github_file_url(url: str) -> dict:
    """Parse a GitHub blob/tree URL into its components.

    Returns a dict with keys: owner, repo, ref, path, start_line, end_line, kind.
    Raises ``ValueError`` if the URL doesn't match the expected pattern.
    """
    url = url.strip()

    # Extract line range before matching (fragment is part of URL text)
    start_line: int | None = None
    end_line: int | None = None
    line_match = _LINE_RANGE_RE.search(url)
    if line_match:
        start_line = int(line_match.group("start"))
        end_str = line_match.group("end")
        end_line = int(end_str) if end_str else start_line
        url = url[: line_match.start()]

    m = _GITHUB_FILE_URL_RE.match(url)
    if not m:
        raise ValueError(f"Not a valid GitHub file URL: {url!r}")

    owner = m.group("owner")
    repo = m.group("repo")
    kind = m.group("kind")
    rest = m.group("rest").strip("/")

    # rest = "<ref>/<path>".  The ref may contain slashes (e.g. feature/foo).
    # We split on the first path component that looks like a real file/directory
    # by trying progressively longer ref prefixes until the remainder is non-empty.
    parts = rest.split("/")
    if len(parts) == 1:
        # Only a ref, no path — e.g. /tree/main
        ref = parts[0]
        path = ""
    else:
        # Default: first segment is ref, rest is path.  This is correct for
        # simple branch names (main, master_next).  For branches with slashes
        # we can't disambiguate perfectly without an API call, so we use the
        # shortest ref (first segment) which covers the vast majority of cases.
        ref = parts[0]
        path = "/".join(parts[1:])

    return {
        "owner": owner,
        "repo": repo,
        "ref": ref,
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "kind": kind,
    }


# ---------------------------------------------------------------------------
# GitHub read tools (for TaskReader)
# ---------------------------------------------------------------------------

_MAX_FILE_CHARS = 50_000


def make_github_read_tools(github_adapter) -> list[ToolDefinition]:
    """Read-only GitHub tools for fetching file contents from URLs.

    Designed for the TaskReader agent so it can read source files
    referenced in Jira ticket descriptions and comments.

    Args:
        github_adapter: GitHubAdapter instance with a valid token.
    """

    def fetch_github_file(url: str) -> str:
        try:
            parsed = _parse_github_file_url(url)
        except ValueError as exc:
            return f"Error: {exc}"
        try:
            content = github_adapter.get_file_contents(
                parsed["owner"],
                parsed["repo"],
                parsed["path"],
                ref=parsed["ref"],
            )
        except Exception as exc:
            return (
                f"Error fetching {parsed['owner']}/{parsed['repo']}/"
                f"{parsed['path']} (ref={parsed['ref']}): {exc}"
            )

        # Apply line-range filter if URL had #L... fragment
        if parsed["start_line"] is not None:
            lines = content.splitlines()
            start = max(0, parsed["start_line"] - 1)  # 1-based → 0-based
            end = parsed["end_line"] or len(lines)
            # Include some surrounding context (10 lines before, 5 after)
            ctx_start = max(0, start - 10)
            ctx_end = min(len(lines), end + 5)
            selected = lines[ctx_start:ctx_end]
            header = (
                f"# {parsed['path']}  "
                f"(lines {ctx_start + 1}-{ctx_end} of {len(lines)})\n"
            )
            content = header + "\n".join(
                f"{ctx_start + i + 1:>5}: {line}"
                for i, line in enumerate(selected)
            )

        if len(content) > _MAX_FILE_CHARS:
            content = content[:_MAX_FILE_CHARS] + "\n... (truncated)"
        return content

    def list_github_directory(url: str) -> str:
        try:
            parsed = _parse_github_file_url(url)
        except ValueError as exc:
            return f"Error: {exc}"
        try:
            all_paths = github_adapter.get_repo_tree(
                parsed["owner"], parsed["repo"], parsed["ref"],
            )
        except Exception as exc:
            return (
                f"Error listing {parsed['owner']}/{parsed['repo']} "
                f"(ref={parsed['ref']}): {exc}"
            )

        prefix = parsed["path"]
        if prefix:
            # Filter to entries under the requested directory
            filtered = []
            for p in all_paths:
                if p.startswith(prefix + "/") or p == prefix:
                    # Show path relative to the requested directory
                    rel = p[len(prefix):].lstrip("/")
                    if rel:
                        filtered.append(rel)
            if not filtered:
                return f"No files found under '{prefix}' on ref '{parsed['ref']}'."
            paths = filtered
        else:
            paths = all_paths

        # Filter ignored directories
        result: list[str] = []
        for p in paths:
            parts = p.split("/")
            if any(part in _IGNORED_DIRS or part.endswith(".egg-info") for part in parts):
                continue
            result.append(p)
        return "\n".join(result[:500]) if result else "(empty)"

    return [
        ToolDefinition(
            name="fetch_github_file",
            description=(
                "Fetch the contents of a file from a GitHub URL. "
                "Accepts full GitHub blob URLs like "
                "'https://github.com/owner/repo/blob/branch/path/to/file.py' "
                "or URLs with line anchors like '...file.py#L10-L25'. "
                "Returns the file content (with line numbers if a range was specified)."
            ),
            parameters=[
                ToolParam("url", "string", "Full GitHub file URL (blob URL)"),
            ],
            handler=fetch_github_file,
        ),
        ToolDefinition(
            name="list_github_directory",
            description=(
                "List files in a GitHub repository directory from a URL. "
                "Accepts full GitHub tree/blob URLs like "
                "'https://github.com/owner/repo/tree/branch/path/to/dir'. "
                "Returns file paths under that directory."
            ),
            parameters=[
                ToolParam("url", "string", "Full GitHub URL (tree or blob URL)"),
            ],
            handler=list_github_directory,
        ),
    ]


# ---------------------------------------------------------------------------
# GitHub API tools (for Reviewer)
# ---------------------------------------------------------------------------

def make_github_review_tools(
    github,
    owner: str,
    repo: str,
    pr_number: int,
) -> list[ToolDefinition]:
    """Tools for reviewing a GitHub pull request.

    Args:
        github: GitHubAdapter instance
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number
    """

    def get_pr_diff() -> str:
        return github.get_pr_diff(owner, repo, pr_number)

    def get_pr_info() -> str:
        pr = github.get_pull_request(owner, repo, pr_number)
        return (
            f"Title: {pr.title}\n"
            f"Author: {pr.author}\n"
            f"Base: {pr.base_ref}\n"
            f"Head: {pr.head_ref}\n"
            f"Body:\n{pr.body}\n"
            f"Additions: {pr.additions}\n"
            f"Deletions: {pr.deletions}\n"
            f"Changed files: {pr.changed_files}"
        )

    def list_repo_files() -> str:
        default_branch = github.get_repo_default_branch(owner, repo)
        paths = github.get_repo_tree(owner, repo, default_branch)
        filtered: list[str] = []
        for path in paths:
            parts = path.split("/")
            if any(p in _IGNORED_DIRS or p.endswith(".egg-info") for p in parts):
                continue
            filtered.append(path)
        return "\n".join(sorted(filtered))

    def read_repo_file(path: str) -> str:
        try:
            return github.get_file_contents(owner, repo, path)
        except Exception as exc:
            return f"Error fetching '{path}': {exc}"

    def get_pr_changed_files() -> str:
        files = github.get_pr_files(owner, repo, pr_number)
        parts: list[str] = []
        for f in files:
            parts.append(
                f"{f.status} {f.filename} "
                f"(+{f.additions} -{f.deletions})"
            )
        return "\n".join(parts) if parts else "No changed files."

    return [
        ToolDefinition(
            name="get_pr_diff",
            description="Get the unified diff for the pull request being reviewed.",
            parameters=[],
            handler=get_pr_diff,
        ),
        ToolDefinition(
            name="get_pr_info",
            description="Get metadata about the pull request (title, author, body, stats).",
            parameters=[],
            handler=get_pr_info,
        ),
        ToolDefinition(
            name="get_pr_changed_files",
            description="List all files changed in the pull request with their status and stats.",
            parameters=[],
            handler=get_pr_changed_files,
        ),
        ToolDefinition(
            name="list_repo_files",
            description="List all files in the repository (default branch).",
            parameters=[],
            handler=list_repo_files,
        ),
        ToolDefinition(
            name="read_repo_file",
            description="Read the contents of a file from the repository (default branch).",
            parameters=[
                ToolParam("path", "string", "Path to the file in the repository"),
            ],
            handler=read_repo_file,
        ),
    ]


# ---------------------------------------------------------------------------
# Jenkins tools (for TaskReader)
# ---------------------------------------------------------------------------

def make_jenkins_tools(jenkins_adapter) -> list[ToolDefinition]:
    """Tools for fetching Jenkins build information.

    Args:
        jenkins_adapter: JenkinsAdapter instance.
    """

    def fetch_jenkins_build(url: str) -> str:
        try:
            build = jenkins_adapter.get_build_info(url)
            duration_s = build.duration_ms / 1000 if build.duration_ms else 0
            parts = [
                f"Build: {build.full_display_name}",
                f"Result: {build.result or 'IN PROGRESS'}",
                f"Duration: {duration_s:.0f}s",
                f"URL: {build.url}",
            ]
            if build.parameters:
                params = "\n".join(f"  {k} = {v}" for k, v in build.parameters.items())
                parts.append(f"Parameters:\n{params}")
            return "\n".join(parts)
        except Exception as exc:
            logger.error("fetch_jenkins_build(%s) failed: %s", url, exc)
            return f"Error fetching Jenkins build info for {url}: {exc}"

    def fetch_jenkins_log(url: str, tail_lines: int = 500) -> str:
        try:
            return jenkins_adapter.get_console_log(url, tail_lines=tail_lines)
        except Exception as exc:
            logger.error("fetch_jenkins_log(%s) failed: %s", url, exc)
            return f"Error fetching Jenkins log for {url}: {exc}"

    def search_jenkins_log(url: str, pattern: str) -> str:
        try:
            return jenkins_adapter.search_console_log(url, pattern)
        except Exception as exc:
            logger.error("search_jenkins_log(%s, %s) failed: %s", url, pattern, exc)
            return f"Error searching Jenkins log for {url}: {exc}"

    def fetch_jenkins_test_report(url: str) -> str:
        try:
            report = jenkins_adapter.get_test_report(url)
            if report is None:
                return f"No test report available for {url}."
            parts = [
                f"Total: {report.total}  Passed: {report.passed}  "
                f"Failed: {report.failed}  Skipped: {report.skipped}",
            ]
            if report.cases:
                parts.append("\nFailed/errored tests:")
                for case in report.cases[:30]:
                    parts.append(f"  - {case.class_name}.{case.name} [{case.status}]")
                    if case.error_message:
                        # Truncate per-case error to keep output manageable
                        msg = case.error_message[:500]
                        parts.append(f"    {msg}")
            return "\n".join(parts)
        except Exception as exc:
            logger.error("fetch_jenkins_test_report(%s) failed: %s", url, exc)
            return f"Error fetching Jenkins test report for {url}: {exc}"

    return [
        ToolDefinition(
            name="fetch_jenkins_build",
            description=(
                "Fetch metadata for a Jenkins build (status, duration, parameters). "
                "Provide the full Jenkins build URL."
            ),
            parameters=[
                ToolParam("url", "string", "Full Jenkins build URL"),
            ],
            handler=fetch_jenkins_build,
        ),
        ToolDefinition(
            name="fetch_jenkins_log",
            description=(
                "Fetch console log output from a Jenkins build. Returns the "
                "last tail_lines lines (default 500). Use search_jenkins_log "
                "first for large logs to find relevant sections."
            ),
            parameters=[
                ToolParam("url", "string", "Full Jenkins build URL"),
                ToolParam(
                    "tail_lines", "integer",
                    "Number of lines from the end to return (default 500, use 0 for full log)",
                    required=False,
                ),
            ],
            handler=fetch_jenkins_log,
        ),
        ToolDefinition(
            name="search_jenkins_log",
            description=(
                "Search a Jenkins build's console log for a pattern. "
                "Returns matching lines with surrounding context. "
                "Use this BEFORE fetch_jenkins_log to find relevant "
                "sections in large logs (e.g. search for 'error', "
                "'failed', 'exception', 'traceback')."
            ),
            parameters=[
                ToolParam("url", "string", "Full Jenkins build URL"),
                ToolParam("pattern", "string", "Text or regex pattern to search for"),
            ],
            handler=search_jenkins_log,
        ),
        ToolDefinition(
            name="fetch_jenkins_test_report",
            description=(
                "Fetch JUnit test report from a Jenkins build. Returns "
                "pass/fail/skip counts and details of failed tests. "
                "Returns 'no test report' if the build has none."
            ),
            parameters=[
                ToolParam("url", "string", "Full Jenkins build URL"),
            ],
            handler=fetch_jenkins_test_report,
        ),
    ]
