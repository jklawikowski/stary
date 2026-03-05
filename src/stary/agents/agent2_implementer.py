"""Agent 2: Implementer – validates tasks against the real repository
structure, aligns to repo architecture and agent-context markdown files,
then calls an LLM to generate file contents, writes them to a fresh clone,
commits and pushes the changes to a feature branch."""

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from stary.inference import BaseInferenceClient, InferenceClient, get_inference_client
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

# Regex to find a GitHub (or generic git) repo URL in free text.
_REPO_URL_RE = re.compile(
    r"https?://[^\s)\"'>\[\]]+\.git\b"       # explicit .git suffix
    r"|https?://github\.com/[^\s)\"'>\[\]]+", # github shorthand (no .git)
    re.IGNORECASE,
)


class ImplementerAgent:
    """Clones the target repository, validates the incoming task list against
    the actual file structure and any agent-context markdown files found in
    the repo, calls the LLM to generate file contents, writes them to the
    clone, then commits and pushes the result to a feature branch."""

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

    def run(self, task_input: dict) -> str:
        """
        End-to-end implementation pipeline.

        Parameters
        ----------
        task_input : dict
            Output of Agent 1 containing at least:
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
        str  – URL to the created pull request.
        """
        # 0. Extract repo info from input ----------------------------------
        repo_url = self._extract_repo_url(task_input)
        ticket_id = task_input.get("ticket_id", "UNKNOWN")
        summary = task_input.get("summary", "feature implementation")
        print(f"[Agent2] Repo URL : {repo_url}")
        print(f"[Agent2] Ticket   : {ticket_id}")

        # 1. Clone the repository ------------------------------------------
        self.repo_path = self._clone_repo(repo_url)
        print(f"[Agent2] Cloned to: {self.repo_path}")

        # 2. Create feature branch -----------------------------------------
        branch_name = self._create_branch(ticket_id)
        print(f"[Agent2] Branch   : {branch_name}")

        # 3. Discover repo layout ------------------------------------------
        repo_tree = self._scan_repo_tree()
        print(f"[Agent2] Scanned repo – {len(repo_tree)} file(s) found.")

        # 4. Read agent-context markdown files (source of truth) -----------
        context_docs = self._read_context_markdowns(repo_tree)
        print(f"[Agent2] Found {len(context_docs)} agent-context doc(s).")

        # 5. Read key source files for deeper context ----------------------
        source_snippets = self._read_key_sources(repo_tree)

        # 6. Validate & align tasks against reality ------------------------
        validated = self._validate_and_align(
            task_input, repo_tree, context_docs, source_snippets,
        )
        if not validated or not validated.get("aligned_tasks"):
            print(
                "[Agent2] WARNING: Task validation FAILED – LLM returned "
                "an empty or invalid response. The raw output was:\n"
                f"  {validated!r}\n"
                "Falling back to original (unvalidated) tasks."
            )
            validated = {
                "validation_notes": "Validation skipped (LLM returned empty response)",
                "aligned_tasks": task_input.get("tasks", []),
            }
        else:
            print("[Agent2] Task validation & alignment done.")
            print(validated)

        # 7. Call LLM to generate file contents ----------------------------
        file_ops = self._generate_files_via_llm(
            task_input, validated, repo_tree, context_docs, source_snippets,
        )
        print(f"[Agent2] LLM produced {len(file_ops)} file operation(s).")

        # 8. Write files to the cloned repo --------------------------------
        self._write_files(file_ops)
        print("[Agent2] Files written successfully.")

        # 9. Commit & push -------------------------------------------------
        commit_msg = f"{ticket_id} {summary}"
        branch_url = self._commit_and_push(commit_msg, branch_name, repo_url)
        print(f"[Agent2] Pushed to: {branch_url}")

        # 10. Create pull request ------------------------------------------
        pr_url = self._create_pull_request(
            repo_url, branch_name, ticket_id, summary,
        )
        print(f"[Agent2] PR created: {pr_url}")

        return pr_url

    # ------------------------------------------------------------------
    # Create pull request
    # ------------------------------------------------------------------

    def _create_pull_request(
        self,
        repo_url: str,
        branch_name: str,
        ticket_id: str,
        summary: str,
        base_branch: str = "master",
    ) -> str:
        """Create a GitHub pull request from *branch_name* to *base_branch*
        and return the PR URL.

        Uses the ``github_token`` provided at construction (or falls back to
        the ``GITHUB_TOKEN`` environment variable) for authentication.
        """
        if not self.github_token:
            raise EnvironmentError(
                "[Agent2] github_token is required to create a pull request. "
                "Pass it to the constructor or set GITHUB_TOKEN env var."
            )

        # Extract owner/repo from the URL
        # e.g. https://github.com/intel-sandbox/calculator -> intel-sandbox/calculator
        url_clean = repo_url.rstrip("/")
        if url_clean.endswith(".git"):
            url_clean = url_clean[:-4]
        parsed = urlparse(url_clean)
        # path looks like /intel-sandbox/calculator
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(
                f"[Agent2] Cannot extract owner/repo from URL: {repo_url}"
            )
        owner, repo = parts[0], parts[1]

        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github+json",
        }
        body = {
            "title": f"{ticket_id} {summary}",
            "head": branch_name,
            "base": base_branch,
            "body": (
                f"Automated implementation for **{ticket_id}**.\n\n"
                f"**Summary:** {summary}\n\n"
                f"Branch: `{branch_name}`"
            ),
        }

        resp = requests.post(api_url, json=body, headers=headers, timeout=60)
        resp.raise_for_status()
        pr_data = resp.json()
        return pr_data["html_url"]

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
            "[Agent2] Could not find a repository URL in the task input. "
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
            print(f"[Agent2] Removing existing clone at {dest}")
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
    # Write generated files
    # ------------------------------------------------------------------

    def _write_files(self, file_ops: list[dict]) -> None:
        """Write file operations produced by the LLM to the cloned repo.

        Each entry in *file_ops* must have:
        - ``path``    : repo-relative file path
        - ``content`` : full file content
        - ``action``  : "create" | "modify" | "delete"  (default: "modify")
        """
        root = Path(self.repo_path)
        for op in file_ops:
            rel = op.get("path", "")
            action = op.get("action", "modify").lower()
            full = root / rel

            if action == "delete":
                if full.exists():
                    full.unlink()
                    print(f"[Agent2]   deleted {rel}")
                continue

            # create / modify → write content
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(op.get("content", ""))
            print(f"[Agent2]   wrote   {rel}")

    # ------------------------------------------------------------------
    # Commit & push
    # ------------------------------------------------------------------

    def _commit_and_push(
        self,
        commit_msg: str,
        branch_name: str,
        repo_url: str,
    ) -> str:
        """Stage all changes, commit, push to origin, and return the branch
        URL on GitHub."""
        run = lambda cmd: subprocess.run(
            cmd,
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )

        # Configure git identity (required for commit in containers)
        run(["git", "config", "user.name", self.git_user_name])
        run(["git", "config", "user.email", self.git_user_email])

        run(["git", "add", "-A"])
        run(["git", "commit", "-m", commit_msg])

        # Use authenticated URL for push if token is available
        if self.github_token and "github.com" in repo_url:
            auth_url = repo_url.replace("https://github.com", f"https://{self.github_token}@github.com")
            run(["git", "push", auth_url, branch_name])
        else:
            run(["git", "push", "origin", branch_name])

        # Build a browsable URL for the branch
        # e.g. https://github.com/org/repo -> https://github.com/org/repo/tree/<branch>
        base = repo_url.rstrip("/")
        if base.endswith(".git"):
            base = base[:-4]
        branch_url = f"{base}/tree/{branch_name}"
        return branch_url

    # ------------------------------------------------------------------
    # 1. Repo scanning
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

    # ------------------------------------------------------------------
    # 2. Agent-context markdown files
    # ------------------------------------------------------------------

    def _read_context_markdowns(self, repo_tree: list[str]) -> dict[str, str]:
        """Find all markdown files in the repo and read them as context docs.

        Every .md file in the repository is considered a source-of-truth
        document (architecture, conventions, agent instructions, specs,
        design docs, exec plans, etc.).
        """
        docs: dict[str, str] = {}
        for rel in repo_tree:
            if _CONTEXT_MD_PATTERN.search(rel):
                full = os.path.join(self.repo_path, rel)
                try:
                    docs[rel] = Path(full).read_text(errors="replace")
                except OSError:
                    pass
        return docs

    # ------------------------------------------------------------------
    # 3. Read key source files
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

        print(f"[Agent2] Read {len(snippets)} source file(s) ({total} bytes).")
        return snippets

    # ------------------------------------------------------------------
    # 4. Validate & align
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
        tree_str = "\n".join(repo_tree)
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
            "- Correct and refine each task so it is grounded in the actual repo structure.\n"
            "- Preserve the intent of each task but rewrite paths, module names, and "
            "technical details so they match reality.\n"
            "- If the context markdowns define conventions or architecture rules, "
            "tasks MUST follow them.\n\n"
            "Return ONLY valid JSON (no markdown fences) with this schema:\n"
            "{\n"
            '  "validation_notes": "<summary of mismatches found and corrections made>",\n'
            '  "aligned_tasks": [\n'
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
    # 5. Generate file contents via LLM
    # ------------------------------------------------------------------

    def _generate_files_via_llm(
        self,
        task_input: dict,
        validated: dict,
        repo_tree: list[str],
        context_docs: dict[str, str],
        source_snippets: dict[str, str],
    ) -> list[dict]:
        """Call the LLM to produce a list of file operations implementing
        each task.  Returns a list of dicts, each with keys ``path``,
        ``content``, and ``action``."""

        tree_str = "\n".join(repo_tree)
        context_block = "\n\n".join(
            f"### {path}\n{content}" for path, content in context_docs.items()
        )
        source_block = "\n\n".join(
            f"### {path}\n```\n{content}\n```"
            for path, content in source_snippets.items()
        )

        aligned_tasks = json.dumps(
            validated.get("aligned_tasks", task_input.get("tasks", [])),
            indent=2,
        )
        validation_notes = validated.get("validation_notes", "")
        implementer_prompt = task_input.get("implementer_prompt", "")

        system = (
            "You are an expert software engineer. Your job is to implement "
            "the requested changes by producing the FULL content of every "
            "file that needs to be created or modified.\n\n"
            "You have NO tools available. Do NOT attempt to call any functions "
            "or tools. Do NOT browse, search, or read files. Work ONLY with "
            "the information provided below.\n\n"
            "CRITICAL RULES — violating ANY of these is a failure:\n"
            "1. Your ENTIRE response must be valid JSON and NOTHING ELSE.\n"
            "2. Do NOT include any commentary, explanation, markdown fences, "
            "follow-up questions, or conversational text.\n"
            "3. Do NOT ask for more information.\n"
            "4. For modified files output the COMPLETE file content (not just "
            "the changed part).\n"
            "5. Follow the architecture, conventions, and naming from the repo "
            "and context docs exactly.\n"
            "6. Implement ALL tasks — do not leave TODOs or placeholders.\n\n"
            "Return ONLY a JSON array (no markdown fences) with this schema:\n"
            "[\n"
            '  {"path": "<repo-relative path>", '
            '"action": "create|modify|delete", '
            '"content": "<full file content>"},\n'
            "  ...\n"
            "]\n"
        )

        user = (
            f"## Implementer prompt from ticket analysis\n{implementer_prompt}\n\n"
            f"## Validated & aligned tasks\n```json\n{aligned_tasks}\n```\n\n"
            f"## Validation notes\n{validation_notes}\n\n"
            f"## Repository file tree\n```\n{tree_str}\n```\n\n"
            f"## Agent-context documents\n{context_block or '(none)'}\n\n"
            f"## Existing source code\n{source_block or '(none)'}\n\n"
            "Generate the JSON array of file operations now."
        )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(
                    f"[Agent2/implement] Attempt {attempt}/{max_attempts} "
                    f"— previous response was not valid JSON, retrying…"
                )
            raw_text = self._call_llm(system, user, tag="implement", raw=True)

            if not raw_text:
                continue

            # Robust JSON extraction — the LLM often wraps JSON in prose
            # or markdown fences.
            parsed = BaseInferenceClient.extract_json_array(raw_text)

            if isinstance(parsed, list) and parsed:
                return parsed

            # Sometimes the LLM wraps the list in an object
            if isinstance(parsed, dict):
                for key in ("files", "file_ops", "operations", "changes"):
                    if isinstance(parsed.get(key), list):
                        return parsed[key]

        raise RuntimeError(
            "[Agent2/implement] LLM failed to produce valid file operations "
            f"after {max_attempts} attempts."
        )

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
        raw: bool = False,
    ) -> Any:
        """Send a chat-completion request via inference client.

        Parameters
        ----------
        raw : bool
            If True, return the raw text content instead of parsing as JSON.
        """
        label = f"[Agent2/{tag}]" if tag else "[Agent2]"
        print(f"{label} Calling inference…")

        try:
            if raw:
                content = self._inference.chat(
                    system=system,
                    user=user,
                    temperature=temperature,
                    timeout=300.0,
                )
                print(f"{label} Got raw response ({len(content)} chars)")
                return content.strip() if content else ""
            else:
                result = self._inference.chat_json(
                    system=system,
                    user=user,
                    temperature=temperature,
                    timeout=300.0,
                )
                print(f"{label} Got JSON response: {result}")
                return result

        except Exception as exc:
            print(f"{label} LLM request failed: {exc}")
            return "" if raw else {}


