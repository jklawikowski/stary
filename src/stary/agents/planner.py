"""Planner – validates tasks against the real repository structure using
tool-calling to actively explore the codebase.

Clones the target repository, then gives the LLM tools to list
directories, read files, and search code.  The LLM autonomously
explores the repo to validate and refine the task list from
TaskReader, producing concrete implementation steps for the
Implementer.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from stary.agents.tools import make_read_tools
from stary.github_adapter import GitHubAdapter
from stary.inference import InferenceClient, ToolDefinition, ToolParam, get_inference_client

PLAYGROUND_DIR = Path.home() / "playground"

_IGNORED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "*.egg-info",
}

_SYSTEM_PROMPT = """\
You are an expert software engineer performing a pre-implementation
validation step.

You have tools to explore a cloned repository. Use them freely to
understand the real project structure before producing your output.

Your workflow:
1. Use list_directory to understand the top-level repo layout.
2. Read key files (README, architecture docs, pyproject.toml, etc.)
   to understand project conventions and structure.
3. Explore directories relevant to the tasks.
4. Read source files that the tasks reference or affect.
5. Based on your exploration, validate and refine the task list.

Your job:
- Check every task for alignment with the actual repository architecture.
- Identify file-path mismatches, missing modules, naming inconsistencies,
  or assumptions that contradict the real code.
- Break the work into discrete, self-contained IMPLEMENTATION STEPS.
  Each step will be sent to an implementer LLM in a SEPARATE call.
- Preserve the intent of each task but rewrite paths, module names, and
  technical details so they match reality.
- If the context markdowns define conventions or architecture rules,
  steps MUST follow them.

## Implementer capabilities — READ THIS CAREFULLY

The implementer LLM has these tools:
- read_file(path): Read a file's contents.
- list_directory(path): List files/subdirectories.
- search_files(pattern, path): Find files matching a pattern.
- search_code(pattern, path): Search file contents with line numbers.
- write_file(path, content): Create or overwrite a file.
- modify_file(path, old_text, new_text): Search-and-replace in a file.
- delete_file(path): Delete a single file.
- run_command(command): Run a shell command in the repo root. Allowed:
  git, find, ls, grep, cat, head, tail, sort, uniq, xargs, echo, test,
  mkdir, rm, cp, mv, touch, dirname, basename, sed, awk, wc.

After all steps finish, git add/commit/push and PR creation are handled
AUTOMATICALLY by the system — do NOT include git commit, git push, or
PR creation instructions in any step.

When writing steps:
- Use run_command for bulk operations (e.g. 'git rm -r <pattern>' to
  untrack many files, 'find ... -exec rm ...' for mass deletion).
- Use modify_file or write_file for targeted file edits.
- NEVER instruct the implementer to run git commit, git push, or
  create a PR — those happen automatically.

Each step MUST have a `prompt` field — a COMPLETE, SELF-CONTAINED
instruction for the implementer LLM. The implementer receives ONLY
this prompt plus tool access to the same repo. The prompt must include:
- Precise description of WHAT to implement — target function/class
  names, parameters, return types, key constraints.
- Any corrections or context from your validation.
- Enough detail that the implementer can produce the code without
  further questions.

Each step MAY optionally include `target_files` — a list of
repo-relative file paths the step will read or modify.

Order steps so earlier steps don't depend on later ones.

## Code style — IMPORTANT

During exploration, READ the repository's linter/formatter config:
- pyproject.toml (look for [tool.ruff], [tool.isort], [tool.black])
- setup.cfg, .flake8, ruff.toml, .pre-commit-config.yaml

Include in EVERY step's prompt:
- The line-length limit the repo uses.
- The import sort style (isort profile, ruff isort rules).
- Any quote-style or trailing-comma conventions.
- An explicit instruction: "All code MUST pass the repo's configured
  linters (e.g. ruff, isort). Do NOT leave unused imports or variables."

If no linter config exists, instruct steps to use: isort-compatible
imports, 88-char line length, double quotes, trailing commas.

When you have finished exploring and are ready, provide your FINAL
answer as ONLY valid JSON (no markdown fences) with this schema:
{
  "steps": [
    {"prompt": "<complete implementer instruction>",
     "target_files": ["path/in/repo", ...]},
    ...
  ]
}
"""


class Planner:
    """Clones the target repository and uses tool-calling to explore it,
    then validates and refines the task list for the Implementer."""

    def __init__(
        self,
        inference_client: InferenceClient | None = None,
        github: GitHubAdapter | None = None,
    ):
        self._inference = inference_client or get_inference_client()
        self._github = github or GitHubAdapter()
        self.repo_path: str | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self, task_input: dict) -> dict:
        """Validate and align tasks against the real repository.

        Parameters
        ----------
        task_input : dict
            Output of TaskReader.

        Returns
        -------
        dict – Context for the Implementer:
            - steps       : list[dict]
            - repo_url    : str
            - repo_path   : str
            - branch_name : str
            - ticket_id   : str
            - summary     : str
        """
        repo_url = self._extract_repo_url(task_input)
        ticket_id = task_input.get("ticket_id", "UNKNOWN")
        summary = task_input.get("summary", "feature implementation")
        logger.info("Repo URL : %s", repo_url)
        logger.info("Ticket   : %s", ticket_id)

        # 1. Clone
        self.repo_path = self._clone_repo(repo_url)
        logger.info("Cloned to: %s", self.repo_path)

        # 2. Create feature branch
        branch_name = self._create_branch(ticket_id)
        logger.info("Branch   : %s", branch_name)

        # 3. Build tools for repo exploration
        tools = make_read_tools(self.repo_path)

        # 4. Call LLM with tools to validate and refine tasks
        tasks_json = json.dumps(task_input.get("tasks", []), indent=2)
        user = (
            f"## Tasks from ticket analysis\n```json\n{tasks_json}\n```\n\n"
            f"## Ticket summary\n{summary}\n\n"
            f"## Implementer prompt from TaskReader\n"
            f"{task_input.get('implementer_prompt', '(none)')}\n\n"
            "Start by exploring the repository structure with list_directory "
            "and reading key files to understand the codebase. Then validate "
            "and refine the tasks above into concrete implementation steps."
        )

        logger.info("Starting tool-calling session for task validation")
        try:
            validated = self._inference.chat_json_with_tools(
                system=_SYSTEM_PROMPT,
                user=user,
                tools=tools,
                temperature=0.2,
                timeout=900.0,
                max_iterations=30,
            )
        except Exception as exc:
            logger.error("LLM failed: %s", exc)
            validated = {}

        # 5. Handle result
        if not validated or not validated.get("steps"):
            logger.warning(
                "Task validation failed. Falling back to original tasks."
            )
            fallback_steps = [
                {
                    "prompt": (
                        f"Implement: {t.get('title', f'Task {i+1}')}\n\n"
                        f"{t.get('detail', '')}"
                    ),
                    "target_files": t.get("target_files", []),
                }
                for i, t in enumerate(task_input.get("tasks", []))
            ]
            validated = {"steps": fallback_steps}
        else:
            logger.info(
                "Validation done — %d step(s)", len(validated["steps"]),
            )

        return {
            "steps": validated["steps"],
            "repo_url": repo_url,
            "repo_path": self.repo_path,
            "branch_name": branch_name,
            "ticket_id": ticket_id,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Repo URL extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_repo_url(task_input: dict) -> str:
        """Return the target repository URL from LLM analysis."""
        repo_url = task_input.get("repo_url", "").strip()
        if not repo_url:
            raise ValueError(
                "[Planner] Could not find a repository URL in the task input. "
                "Ensure the ticket description contains a GitHub URL."
            )
        return Planner._normalise_github_url(repo_url)

    @staticmethod
    def _normalise_github_url(url: str) -> str:
        """Normalise a GitHub URL to ``https://github.com/<owner>/<repo>``.

        Handles deep links like ``/tree/main/src/...`` or ``/blob/dev/...``
        and strips trailing slashes, ``.git`` suffixes, query strings, and
        fragment identifiers.
        """
        from urllib.parse import urlparse

        url = url.strip().rstrip("/")
        parsed = urlparse(url)
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) < 2:
            return url  # Can't normalise — return as-is
        owner, repo = parts[0], parts[1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        return f"https://github.com/{owner}/{repo}"

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def _clone_repo(self, repo_url: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(repo_url)
        repo_name = Path(parsed.path).stem
        dest = PLAYGROUND_DIR / repo_name
        return self._github.clone_repo(repo_url, dest)

    # ------------------------------------------------------------------
    # Branch management
    # ------------------------------------------------------------------

    def _create_branch(self, ticket_id: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        branch_name = f"dev/sys_qaplatformbot/{ticket_id}{ts}"
        return self._github.create_branch(self.repo_path, branch_name)
