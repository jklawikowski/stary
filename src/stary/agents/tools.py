"""Shared tool factories for agents.

Each factory returns a list of ``ToolDefinition`` objects with handlers
bound to a specific context (repo path, Jira adapter, GitHub token, etc.).

Security: all filesystem tools validate paths stay within the repo root.
"""

from __future__ import annotations

import base64
import fnmatch
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import requests

from stary.inference.base import ToolDefinition, ToolParam

# Directories to skip when scanning repos
_IGNORED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs",
}


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

    def fetch_ticket(issue_key: str) -> str:
        try:
            issue = jira_adapter.get_issue(
                issue_key, fields=["summary", "description", "status", "issuetype"],
            )
            return (
                f"Key: {issue.key}\n"
                f"Summary: {issue.summary}\n"
                f"Description:\n{issue.description}"
            )
        except Exception as exc:
            print(f"[JiraTool] fetch_ticket({issue_key}) failed: {exc}")
            return f"Error fetching ticket {issue_key}: {exc}"

    def get_comments(issue_key: str) -> str:
        try:
            comments = jira_adapter.get_comments(issue_key)
            if not comments:
                return "No comments on this ticket."
            parts: list[str] = []
            for c in comments[:20]:
                author = c.get("author", {}).get("displayName", "Unknown")
                body = c.get("body", "")
                parts.append(f"[{author}]: {body[:500]}")
            return "\n---\n".join(parts)
        except Exception as exc:
            return f"Error fetching comments for {issue_key}: {exc}"

    def search_issues(jql: str) -> str:
        try:
            results = jira_adapter.search_issues(jql, max_results=10)
            if not results:
                return "No matching issues found."
            parts: list[str] = []
            for issue in results:
                parts.append(f"{issue.key}: {issue.summary}")
            return "\n".join(parts)
        except Exception as exc:
            return f"Error searching Jira: {exc}"

    return [
        ToolDefinition(
            name="fetch_ticket",
            description="Fetch a Jira ticket by its issue key (e.g. PROJ-123). Returns summary, description, and metadata.",
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
            name="search_issues",
            description="Search Jira issues with a JQL query.",
            parameters=[
                ToolParam("jql", "string", "JQL query string"),
            ],
            handler=search_issues,
        ),
    ]


# ---------------------------------------------------------------------------
# GitHub API tools (for Reviewer)
# ---------------------------------------------------------------------------

def make_github_review_tools(
    github_token: str,
    owner: str,
    repo: str,
    pr_number: int,
) -> list[ToolDefinition]:
    """Tools for reviewing a GitHub pull request."""

    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    def get_pr_diff() -> str:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        diff_headers = {**headers, "Accept": "application/vnd.github.v3.diff"}
        resp = requests.get(url, headers=diff_headers, timeout=60)
        resp.raise_for_status()
        return resp.text

    def get_pr_info() -> str:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return (
            f"Title: {data.get('title', '')}\n"
            f"Author: {data.get('user', {}).get('login', '')}\n"
            f"Base: {data.get('base', {}).get('ref', '')}\n"
            f"Head: {data.get('head', {}).get('ref', '')}\n"
            f"Body:\n{data.get('body', '')}\n"
            f"Additions: {data.get('additions', 0)}\n"
            f"Deletions: {data.get('deletions', 0)}\n"
            f"Changed files: {data.get('changed_files', 0)}"
        )

    def list_repo_files() -> str:
        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        resp = requests.get(repo_url, headers=headers, timeout=30)
        resp.raise_for_status()
        default_branch = resp.json().get("default_branch", "main")

        tree_url = (
            f"https://api.github.com/repos/{owner}/{repo}"
            f"/git/trees/{default_branch}?recursive=1"
        )
        resp = requests.get(tree_url, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        paths: list[str] = []
        for item in data.get("tree", []):
            if item.get("type") != "blob":
                continue
            path = item["path"]
            parts = path.split("/")
            if any(p in _IGNORED_DIRS or p.endswith(".egg-info") for p in parts):
                continue
            paths.append(path)
        return "\n".join(sorted(paths))

    def read_repo_file(path: str) -> str:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("encoding") == "base64":
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            dl = data.get("download_url")
            if dl:
                return requests.get(dl, timeout=30).text
        except Exception as exc:
            return f"Error fetching '{path}': {exc}"
        return f"Could not read '{path}'."

    def get_pr_changed_files() -> str:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        files = resp.json()
        parts: list[str] = []
        for f in files:
            parts.append(
                f"{f.get('status', '?')} {f.get('filename', '?')} "
                f"(+{f.get('additions', 0)} -{f.get('deletions', 0)})"
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
