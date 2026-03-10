"""Planner – validates tasks against the real repository structure,
aligns to repo architecture and agent-context markdown files.

Clones the target repository, scans its layout, reads context docs and
source files, then calls an LLM to validate and align the task list
produced by TaskReader.  Passes the enriched context to the Implementer.
"""

import json
import os
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from stary.inference import InferenceClient, get_inference_client

PLAYGROUND_DIR = Path.home() / "playground"

# Directories / patterns to skip while scanning the repo tree
_IGNORED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "*.egg-info",
}

# All .md files in the repo are treated as context documents (architecture,
# conventions, agent instructions, design docs, etc.).
_CONTEXT_MD_PATTERN = re.compile(r"\.md$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Prompt budget constants – keep total prompt under model context limits.
# Characters are a rough proxy (~4 chars ≈ 1 token).  200 K chars ≈ 50 K
# tokens which is safe for most models and leaves room for the response.
# ---------------------------------------------------------------------------
_MAX_PROMPT_CHARS = 200_000
_TREE_BUDGET      = 30_000   # repo file-tree listing
_CONTEXT_BUDGET   = 40_000   # markdown docs (architecture, conventions, …)
_SOURCE_BUDGET    = 80_000   # actual source code

# Markdown files to deprioritise (auto-generated, noisy)
_LOW_PRIORITY_MD = re.compile(
    r"(changelog|changes|history|migration|release|license|licence|authors"
    r"|contributors|code.of.conduct)",
    re.IGNORECASE,
)

# Regex to find a GitHub (or generic git) repo URL in free text.
_REPO_URL_RE = re.compile(
    r"https?://[^\s)\"'>\[\]]+\.git\b"       # explicit .git suffix
    r"|https?://github\.com/[^\s)\"'>\[\]]+", # github shorthand (no .git)
    re.IGNORECASE,
)


class Planner:
    """Clones the target repository, validates the incoming task list against
    the actual file structure and any agent-context markdown files found in
    the repo, then passes enriched context to the Implementer."""

    def __init__(
        self,
        inference_client: InferenceClient | None = None,
        github_token: str | None = None,
        git_user_name: str | None = None,
        git_user_email: str | None = None,
    ):
        self._inference = inference_client or get_inference_client()
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        self.git_user_name = git_user_name or os.environ.get("GIT_USER_NAME", "qaplatformbot")
        self.git_user_email = git_user_email or os.environ.get("GIT_USER_EMAIL", "sys_qaplatformbot@intel.com")
        self.repo_path: str | None = None  # set after clone

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self, task_input: dict) -> dict:
        """Validate and align tasks against the real repository.

        Parameters
        ----------
        task_input : dict
            Output of TaskReader containing at least:
            - ticket_id          : str
            - summary            : str
            - description        : str   (must contain the repo URL)
            - interpretation     : str
            - tasks              : list[dict]  (each has 'title' and 'detail')
            - implementer_prompt : str
            Optionally:
            - repo_url           : str   (explicit repo URL; falls back to
                                          extracting from description)

        Returns
        -------
        dict  – Lean context for the Implementer (no raw task_input or
        bulk source_snippets – those are redundant once tasks are
        validated and the Implementer reads files on demand per step):
            - steps            : list[dict]  discrete implementation steps
            - validation_notes : str
            - repo_url         : str
            - repo_path        : str    (local clone path)
            - branch_name      : str
            - repo_tree        : list[str]
            - context_docs     : dict[str, str]
            - ticket_id        : str
            - summary          : str
        """
        # 0. Extract repo info from input ----------------------------------
        repo_url = self._extract_repo_url(task_input)
        ticket_id = task_input.get("ticket_id", "UNKNOWN")
        summary = task_input.get("summary", "feature implementation")
        print(f"[Planner] Repo URL : {repo_url}")
        print(f"[Planner] Ticket   : {ticket_id}")

        # 1. Clone the repository ------------------------------------------
        self.repo_path = self._clone_repo(repo_url)
        print(f"[Planner] Cloned to: {self.repo_path}")

        # 2. Create feature branch -----------------------------------------
        branch_name = self._create_branch(ticket_id)
        print(f"[Planner] Branch   : {branch_name}")

        # 3. Discover repo layout ------------------------------------------
        repo_tree = self._scan_repo_tree()
        print(f"[Planner] Scanned repo – {len(repo_tree)} file(s) found.")

        # 4. Read agent-context markdown files (source of truth) -----------
        context_docs = self._read_context_markdowns(repo_tree)
        print(f"[Planner] Found {len(context_docs)} agent-context doc(s).")

        # 5. Read key source files for deeper context ----------------------
        source_snippets = self._read_key_sources(repo_tree)

        # 6. Validate & align tasks against reality ------------------------
        validated = self._validate_and_align(
            task_input, repo_tree, context_docs, source_snippets,
        )
        if not validated or not validated.get("steps"):
            print(
                "[Planner] WARNING: Task validation FAILED – LLM returned "
                "an empty or invalid response. The raw output was:\n"
                f"  {validated!r}\n"
                "Falling back to original (unvalidated) tasks."
            )
            # Convert raw tasks into the step format the Implementer expects
            fallback_steps = [
                {
                    "title": t.get("title", f"Task {i+1}"),
                    "detail": t.get("detail", ""),
                    "target_files": t.get("target_files", []),
                }
                for i, t in enumerate(task_input.get("tasks", []))
            ]
            validated = {
                "validation_notes": "Validation skipped (LLM returned empty response)",
                "steps": fallback_steps,
            }
        else:
            print("[Planner] Task validation & alignment done.")
            print(validated)

        # 7. Resolve directory paths in target_files to real files ----------
        resolved_steps = self._resolve_target_files(
            validated["steps"], repo_tree,
        )
        validated["steps"] = resolved_steps

        return {
            "steps": validated["steps"],
            "validation_notes": validated.get("validation_notes", ""),
            "repo_url": repo_url,
            "repo_path": self.repo_path,
            "branch_name": branch_name,
            "repo_tree": repo_tree,
            "context_docs": context_docs,
            "ticket_id": ticket_id,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Repo URL extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_repo_url(task_input: dict) -> str:
        """Find a git repository URL in the task input.

        Checks (in order):
        1. An explicit ``repo_url`` key.
        2. The ``description`` field (free text from the Jira ticket).
        3. The ``implementer_prompt`` field.

        Raises ``ValueError`` if no URL can be found.
        """
        # 1. Explicit key
        explicit = task_input.get("repo_url", "").strip()
        if explicit:
            return explicit

        # 2. Search in description and implementer_prompt
        for field in ("description", "implementer_prompt", "interpretation"):
            text = task_input.get(field, "")
            match = _REPO_URL_RE.search(text)
            if match:
                url = match.group(0).rstrip("/")
                return url

        raise ValueError(
            "[Planner] Could not find a repository URL in the task input. "
            "Ensure the ticket description contains a GitHub URL or pass "
            "'repo_url' explicitly."
        )

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def _clone_repo(self, repo_url: str) -> str:
        """Clone *repo_url* into ``~/playground/<repo_name>`` and return the
        local path.  If the directory already exists it is removed first so
        we always start from a clean state."""
        parsed = urlparse(repo_url)
        repo_name = Path(parsed.path).stem  # e.g. "calculator"
        dest = PLAYGROUND_DIR / repo_name

        if dest.exists():
            print(f"[Planner] Removing existing clone at {dest}")
            shutil.rmtree(dest)

        PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)

        # Use GITHUB_TOKEN for authentication if available (required for private repos)
        if self.github_token and "github.com" in repo_url:
            # Inject token into URL: https://github.com/... -> https://{token}@github.com/...
            auth_url = repo_url.replace("https://github.com", f"https://{self.github_token}@github.com")
        else:
            auth_url = repo_url

        subprocess.run(
            ["git", "clone", auth_url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        return str(dest)

    # ------------------------------------------------------------------
    # Branch management
    # ------------------------------------------------------------------

    def _create_branch(self, ticket_id: str) -> str:
        """Create and switch to a feature branch.

        Branch name: ``dev/sys_qaplatformbot/<ticket_id><timestamp>``
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        branch_name = f"dev/sys_qaplatformbot/{ticket_id}{ts}"
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return branch_name

    # ------------------------------------------------------------------
    # Repo scanning
    # ------------------------------------------------------------------

    def _scan_repo_tree(self) -> list[str]:
        """Return a list of repo-relative file paths."""
        paths: list[str] = []
        root = Path(self.repo_path)
        for dirpath, dirnames, filenames in os.walk(root):
            # prune ignored directories in-place
            dirnames[:] = [
                d for d in dirnames
                if d not in _IGNORED_DIRS and not d.endswith(".egg-info")
            ]
            for fname in filenames:
                full = Path(dirpath) / fname
                paths.append(str(full.relative_to(root)))
        return sorted(paths)

    @staticmethod
    def _build_tree_string(
        repo_tree: list[str],
        budget: int = _TREE_BUDGET,
    ) -> str:
        """Build a repo tree string that fits within *budget* characters.

        For small repos the full listing is returned.  For large repos the
        output is a compact directory summary showing file counts per
        directory, which is far more token-efficient than listing every
        path.
        """
        full = "\n".join(repo_tree)
        if len(full) <= budget:
            return full

        # Fall back to directory summary
        dir_counts: Counter[str] = Counter()
        for p in repo_tree:
            parts = p.split("/")
            # Count at depth 2 (e.g. "src/models/") or shallower
            key = "/".join(parts[:min(3, len(parts) - 1)]) + "/" if len(parts) > 1 else "(root)"
            dir_counts[key] += 1

        lines: list[str] = []
        total_chars = 0
        for d, cnt in dir_counts.most_common():
            line = f"{d}  ({cnt} file{'s' if cnt != 1 else ''})"
            if total_chars + len(line) + 1 > budget:
                lines.append(f"... ({len(dir_counts) - len(lines)} more directories)")
                break
            lines.append(line)
            total_chars += len(line) + 1

        return "\n".join(sorted(lines))

    # ------------------------------------------------------------------
    # Agent-context markdown files
    # ------------------------------------------------------------------

    def _read_context_markdowns(
        self,
        repo_tree: list[str],
        budget: int = _CONTEXT_BUDGET,
    ) -> dict[str, str]:
        """Read markdown context docs within a character budget.

        Prioritises root-level and architecture-relevant docs over
        deeply nested or auto-generated ones (changelogs, license, etc.).
        """
        md_paths = [r for r in repo_tree if _CONTEXT_MD_PATTERN.search(r)]

        # Sort: root-level first, then by depth, deprioritise noisy files
        def _sort_key(p: str) -> tuple:
            depth = p.count("/")
            is_low = 1 if _LOW_PRIORITY_MD.search(p) else 0
            return (is_low, depth, p)

        md_paths.sort(key=_sort_key)

        docs: dict[str, str] = {}
        total = 0
        for rel in md_paths:
            full = os.path.join(self.repo_path, rel)
            try:
                text = Path(full).read_text(errors="replace")
            except OSError:
                continue
            if total + len(text) > budget:
                remaining = budget - total
                if remaining > 200:
                    docs[rel] = text[:remaining] + "\n... (truncated)"
                    total = budget
                break
            docs[rel] = text
            total += len(text)

        return docs

    # ------------------------------------------------------------------
    # Read key source files
    # ------------------------------------------------------------------

    def _read_key_sources(
        self,
        repo_tree: list[str],
        max_files: int = 30,
        max_bytes: int = 80_000,
    ) -> dict[str, str]:
        """Read relevant source files so the LLM has real code context.

        Prioritises Python files, configs, and __init__.py files.
        """
        priority: list[str] = []
        rest: list[str] = []

        for rel in repo_tree:
            lower = rel.lower()
            if lower.endswith(("__init__.py", "setup.py", "setup.cfg",
                               "pyproject.toml", "readme.md")):
                priority.append(rel)
            elif lower.endswith((".py", ".ts", ".js", ".yaml", ".yml",
                                 ".json", ".toml", ".cfg")):
                rest.append(rel)

        candidates = priority + rest
        snippets: dict[str, str] = {}
        total = 0
        for rel in candidates[:max_files]:
            full = os.path.join(self.repo_path, rel)
            try:
                text = Path(full).read_text(errors="replace")
                if total + len(text) > max_bytes:
                    text = text[: max_bytes - total] + "\n... (truncated)"
                snippets[rel] = text
                total += len(text)
                if total >= max_bytes:
                    break
            except OSError:
                pass

        print(f"[Planner] Read {len(snippets)} source file(s) ({total} bytes).")
        return snippets

    # ------------------------------------------------------------------
    # Validate & align
    # ------------------------------------------------------------------

    def _validate_and_align(
        self,
        task_input: dict,
        repo_tree: list[str],
        context_docs: dict[str, str],
        source_snippets: dict[str, str],
    ) -> dict:
        """Ask the LLM to validate the task list against the repo reality.

        Returns a JSON dict with aligned/corrected tasks.
        """
        tree_str = self._build_tree_string(repo_tree)
        context_block = "\n\n".join(
            f"### {path}\n{content}" for path, content in context_docs.items()
        )
        source_block = "\n\n".join(
            f"### {path}\n```\n{content}\n```"
            for path, content in source_snippets.items()
        )
        tasks_json = json.dumps(task_input.get("tasks", []), indent=2)

        system = (
            "You are an expert software engineer performing a pre-implementation "
            "validation step.\n\n"
            "You have NO tools available. Do NOT attempt to call any functions "
            "or tools. Do NOT browse, search, or read files. Work ONLY with "
            "the information provided below.\n\n"
            "You are given:\n"
            "1. A list of tasks produced by a ticket-analysis agent.\n"
            "2. The real repository file tree.\n"
            "3. Agent-context markdown documents from the repository (source of truth).\n"
            "4. Key source file contents from the repository.\n\n"
            "Your job:\n"
            "- Check every task for alignment with the actual repository architecture.\n"
            "- Identify file-path mismatches, missing modules, naming inconsistencies, "
            "or assumptions that contradict the real code.\n"
            "- Break the work into discrete, self-contained IMPLEMENTATION STEPS.\n"
            "  Each step will be sent to an implementer LLM in a SEPARATE call, "
            "  so every step must be independently understandable.\n"
            "- Preserve the intent of each task but rewrite paths, module names, and "
            "technical details so they match reality.\n"
            "- If the context markdowns define conventions or architecture rules, "
            "steps MUST follow them.\n\n"
            "IMPORTANT — keep output compact. Another LLM will consume each step "
            "as a separate prompt, so verbosity wastes context budget:\n"
            "- `validation_notes`: 2-3 sentences max. Only mention corrections "
            "actually made.\n"
            "- Each step `detail`: precise bullet points describing WHAT to "
            "implement — target function/class names, parameters, return types, "
            "key constraints, and the specific change. Enough detail that an "
            "implementer can produce the code without further questions.\n"
            "- `target_files`: list EVERY repo-relative path the step will "
            "read or modify — the implementer reads ONLY these files.\n"
            "  CRITICAL: every entry in `target_files` MUST be a concrete FILE "
            "  path that appears in the repository file tree provided below. "
            "  NEVER use directory paths (e.g. 'src/', 'scripts/'). If a step "
            "  touches multiple files in a directory, list each file explicitly.\n"
            "- Order steps so earlier steps don't depend on later ones.\n\n"
            "Return ONLY valid JSON (no markdown fences) with this schema:\n"
            "{\n"
            '  "validation_notes": "<brief summary of corrections>",\n'
            '  "steps": [\n'
            '    {"title": "...", "detail": "...", "target_files": ["path/in/repo", ...]},\n'
            "    ...\n"
            "  ]\n"
            "}"
        )

        user = (
            f"## Tasks from ticket analysis\n```json\n{tasks_json}\n```\n\n"
            f"## Repository file tree\n```\n{tree_str}\n```\n\n"
            f"## Agent-context documents\n{context_block or '(none found)'}\n\n"
            f"## Key source files\n{source_block or '(none read)'}"
        )

        return self._call_llm(system, user, tag="validate")

    # ------------------------------------------------------------------
    # Post-validation: resolve directory paths → actual files
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_target_files(
        steps: list[dict],
        repo_tree: list[str],
    ) -> list[dict]:
        """Expand directory prefixes in ``target_files`` to concrete file
        paths from *repo_tree*.  Entries that already match a file in the
        tree are kept as-is.  Unknown paths that don't match any file or
        prefix are dropped with a warning."""
        tree_set = set(repo_tree)

        for step in steps:
            raw_targets: list[str] = step.get("target_files", [])
            resolved: list[str] = []
            for entry in raw_targets:
                entry = entry.rstrip("/")
                if entry in tree_set:
                    # Exact file match — keep
                    resolved.append(entry)
                else:
                    # Treat as directory prefix and expand
                    prefix = entry + "/" if not entry.endswith("/") else entry
                    matched = [f for f in repo_tree if f.startswith(prefix)]
                    if matched:
                        print(
                            f"[Planner] Expanded directory '{entry}/' → "
                            f"{len(matched)} file(s)"
                        )
                        resolved.extend(matched)
                    else:
                        print(
                            f"[Planner] WARNING: target_files entry '{entry}' "
                            f"matches no file or directory — dropped"
                        )

            # Deduplicate while preserving order
            seen: set[str] = set()
            deduped: list[str] = []
            for f in resolved:
                if f not in seen:
                    seen.add(f)
                    deduped.append(f)
            step["target_files"] = deduped

        return steps

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        system: str,
        user: str,
        *,
        tag: str = "",
        temperature: float = 0.2,
    ) -> dict:
        """Send a chat-completion request via inference client and parse JSON."""
        label = f"[Planner/{tag}]" if tag else "[Planner]"
        print(f"{label} Calling inference…")

        try:
            result = self._inference.chat_json(
                system=system,
                user=user,
                temperature=temperature,
                timeout=900.0,
            )
            print(f"{label} Got JSON response: {result}")
            return result

        except Exception as exc:
            print(f"{label} LLM request failed: {exc}")
            return {}
