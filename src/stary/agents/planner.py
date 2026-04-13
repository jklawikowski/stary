"""Planner – validates tasks against the real repository structure using
tool-calling to actively explore the codebase.

Clones the target repository, then gives the LLM tools to list
directories, read files, and search code.  The LLM autonomously
explores the repo to validate and refine the task list from
TaskReader, producing concrete implementation steps for the
Implementer.

The Planner operates on a single repository at a time.  The
Orchestrator is responsible for grouping tasks by repo and calling
the Planner once per repo.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from stary.agents.tools import make_read_tools
from stary.github_adapter import GitHubAdapter
from stary.inference import InferenceClient, get_inference_client

PLAYGROUND_DIR = Path.home() / "playground"

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

## VerifAI / XPU domain guidance

When tasks involve VerifAI or XPU work, pay special attention to:

- **JSON workload configs**: Files with fields like `exec_bin`,
  `workload_base_cmd`, `workload_params`, `pre_actions`, `output_dir`,
  `extra_tests_path`. Steps that modify these should specify exact field
  names and values.
- **XPU adaptations**: When adapting configs from CPU/CUDA to XPU,
  steps should include: changing device type, adjusting tensor parallel
  (TP) size for memory constraints, adding env vars like
  `PYTORCH_ENABLE_XPU_FALLBACK=1`, and setting up XPU-specific
  `pre_actions` (conda env activation, environment variable exports).
- **Test naming**: VerifAI tests follow the convention
  `test__inference__<model>__<config>__<precision>__<mode>__<device>__<backend>`.
  Steps should preserve this convention.
- **Shell scripts**: `run.sh` scripts configure test execution. Steps
  modifying these should be precise about which variables or commands
  to change.

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
    then validates and refines the task list for the Implementer.

    Operates on a single repository.  The Orchestrator groups tasks by
    repo and calls ``run()`` once per repo.
    """

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
            Must contain:
                repo_url  – the GitHub repo URL for this group.
                tasks     – list of task dicts for this repo.
                ticket_id – Jira ticket key.
                summary   – ticket summary string.

        Returns
        -------
        dict – Context for the Implementer:
            steps, repo_url, repo_path, branch_name, ticket_id,
            summary, fork_owner.
        """
        repo_url = self._normalise_github_url(task_input["repo_url"])
        ticket_id = task_input.get("ticket_id", "UNKNOWN")
        summary = task_input.get("summary", "feature implementation")
        logger.info("Repo URL : %s", repo_url)
        logger.info("Ticket   : %s", ticket_id)

        # 1. Check push permissions and fork if necessary
        owner, repo_name_parsed = GitHubAdapter.parse_repo_url(repo_url)

        # Early allowlist check — fail fast before any network/disk I/O
        if self._github._repo_allowlist is not None:
            self._github._repo_allowlist.assert_allowed(owner, repo_name_parsed)

        if self._github.can_push(owner, repo_name_parsed):
            fork_owner = None
            clone_url = repo_url
            logger.info("Direct push access confirmed for %s/%s", owner, repo_name_parsed)
        else:
            logger.info(
                "No push access to %s/%s — forking", owner, repo_name_parsed,
            )
            clone_url = self._github.fork_repo(owner, repo_name_parsed)
            fork_owner = self._github.get_authenticated_user()
            default_branch = self._github.get_repo_default_branch(
                fork_owner, repo_name_parsed,
            )
            self._github.sync_fork(fork_owner, repo_name_parsed, default_branch)
            logger.info("Using fork: %s/%s", fork_owner, repo_name_parsed)

        # 2. Clone
        self.repo_path = self._clone_repo(clone_url)
        logger.info("Cloned to: %s", self.repo_path)

        # 3. Create feature branch
        branch_name = self._create_branch(ticket_id)
        logger.info("Branch   : %s", branch_name)

        # 4. Build tools for repo exploration
        tools = make_read_tools(self.repo_path)

        # 5. Call LLM with tools to validate and refine tasks
        tasks_json = json.dumps(task_input.get("tasks", []), indent=2)
        user = (
            f"## Tasks from ticket analysis\n```json\n{tasks_json}\n```\n\n"
            f"## Ticket summary\n{summary}\n\n"
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
                timeout=1800.0,
                max_iterations=30,
            )
        except Exception as exc:
            logger.error("LLM failed: %s", exc)
            validated = {}

        # 6. Handle result
        if not validated or not validated.get("steps"):
            logger.warning(
                "Task validation failed. Falling back to original tasks.",
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
            "fork_owner": fork_owner,
        }

    # ------------------------------------------------------------------
    # URL normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_github_url(url: str) -> str:
        """Normalise a GitHub URL to ``https://github.com/<owner>/<repo>``.

        Handles deep links like ``/tree/main/src/...`` or ``/blob/dev/...``
        and strips trailing slashes, ``.git`` suffixes, query strings, and
        fragment identifiers.
        """
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
        parsed = urlparse(repo_url)
        repo_name = Path(parsed.path).stem
        dest = PLAYGROUND_DIR / repo_name
        return self._github.clone_repo(repo_url, dest)

    # ------------------------------------------------------------------
    # Branch management
    # ------------------------------------------------------------------

    def _create_branch(self, ticket_id: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        branch_name = f"dev/sys_qaplatformbot/{ticket_id}--{ts}"
        return self._github.create_branch(self.repo_path, branch_name)
