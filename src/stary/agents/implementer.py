"""Implementer -- gives the LLM tools to read and write files directly,
then commits, pushes, and creates a pull request.

One tool-calling session is run per implementation step.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

from stary.agents.tools import make_read_tools, make_shell_tools, make_write_tools
from stary.github_adapter import GitHubAdapter
from stary.inference import InferenceClient, get_inference_client

_STEP_SYSTEM_PROMPT = """\
You are an expert software engineer. Your job is to implement ONE
specific step of a feature by directly editing files in the repository.

You have tools to:
- read_file: Read the contents of any file in the repo.
- list_directory: List files and folders in a directory.
- search_files: Find files matching a pattern.
- search_code: Search for patterns in file contents (with line numbers).
- write_file: Create a new file or overwrite an existing file.
- modify_file: Apply a search/replace edit to an existing file.
  You MUST provide old_text that exactly matches the current file content.
  Use read_file first to see current content. Include 2-3 lines of
  unchanged context around the target.
- delete_file: Delete a file.
- run_command: Run a shell command in the repo root. Allowed commands:
  git, find, ls, grep, cat, head, tail, sort, uniq, xargs, echo, test,
  mkdir, rm, cp, mv, touch, dirname, basename, sed, awk, wc.
  Use this for bulk operations (e.g. 'git rm -r pattern', 'find ... -delete'),
  piped commands, or anything that would be impractical file-by-file.

Your workflow:
1. Read the files relevant to this step to understand current code.
2. Plan your changes.
3. Implement the changes using write_file/modify_file/run_command tools.
4. Verify by reading modified files if needed.

CRITICAL RULES:
- Implement the step FULLY -- no TODOs, no placeholders, no stubs.
- Follow the architecture, conventions, and naming from the existing code.
- Do NOT ask for more information -- use the tools to find what you need.
- Do NOT run git commit, git push, or create PRs -- that is handled
  automatically after all steps finish.
- When done, respond with a brief summary of what you implemented.

## Code style — IMPORTANT

Before writing or modifying ANY file, check the repository for linter /
formatter configuration:
- pyproject.toml (look for [tool.ruff], [tool.isort], [tool.black], etc.)
- setup.cfg, .flake8, .pylintrc, ruff.toml, .pre-commit-config.yaml

You MUST follow whatever style rules the repository defines. In particular:
- Respect import sorting order (isort / ruff isort rules).
- Respect line-length limits.
- Do NOT leave unused imports or variables.
- Match the existing quote style (single vs double).
- Match trailing-comma conventions.
- If no config is found, default to: isort-compatible imports, 88-char
  line length, double quotes, trailing commas in multi-line structures.

An automated formatter will run after all steps, but you should still
produce clean code — the formatter cannot fix logical lint errors like
unused imports or undefined names.
"""


class Implementer:
    """Iterates over implementation steps from the Planner, giving the
    LLM direct file-editing tools for each step.  After all steps,
    commits, pushes, and creates a pull request."""

    def __init__(
        self,
        inference_client: InferenceClient | None = None,
        github: GitHubAdapter | None = None,
    ):
        self._inference = inference_client or get_inference_client()
        self._github = github or GitHubAdapter()
        self.repo_path: str | None = None

    def run(self, planner_output: dict) -> str:
        """End-to-end implementation pipeline.  Returns PR URL."""
        repo_url = planner_output["repo_url"]
        self.repo_path = planner_output["repo_path"]
        branch_name = planner_output["branch_name"]
        ticket_id = planner_output["ticket_id"]
        summary = planner_output["summary"]
        steps = planner_output["steps"]
        fork_owner = planner_output.get("fork_owner")

        logger.info("Ticket   : %s", ticket_id)
        logger.info("Repo     : %s", repo_url)
        logger.info("Branch   : %s", branch_name)
        logger.info("Steps    : %d", len(steps))
        if fork_owner:
            logger.info("Fork     : %s (cross-repo PR)", fork_owner)

        # Build tools for this repo
        read_tools = make_read_tools(self.repo_path)
        write_tools = make_write_tools(self.repo_path)
        shell_tools = make_shell_tools(self.repo_path)
        all_tools = read_tools + write_tools + shell_tools

        for idx, step in enumerate(steps, 1):
            label = step.get("prompt", "")[:60]
            logger.info("--- Step %d/%d: %s... ---", idx, len(steps), label)
            self._implement_step(step, idx, len(steps), all_tools)

        logger.info("All %d step(s) done", len(steps))

        self._auto_lint(self.repo_path)

        commit_msg = f"{ticket_id} {summary}"
        branch_url = self._commit_and_push(
            commit_msg, branch_name, repo_url, fork_owner=fork_owner,
        )
        logger.info("Pushed to: %s", branch_url)

        pr_url = self._create_pull_request(
            repo_url, branch_name, ticket_id, summary, fork_owner=fork_owner,
        )
        logger.info("PR created: %s", pr_url)
        return pr_url

    def _implement_step(self, step: dict, step_idx: int, total_steps: int, tools: list) -> None:
        """Run one tool-calling session for a single implementation step."""
        step_prompt = step.get("prompt", "")

        target_hint = ""
        target_files = step.get("target_files", [])
        if target_files:
            target_hint = (
                f"\n\nHint: This step likely involves these files: "
                f"{', '.join(target_files)}\n"
                "Start by reading them."
            )

        user = (
            f"## Implementation instruction (step {step_idx} of {total_steps})\n"
            f"{step_prompt}{target_hint}\n\n"
            "Use the tools to read relevant files, then implement the changes."
        )

        logger.info("Step %d: starting tool-calling session", step_idx)
        try:
            response = self._inference.chat_with_tools(
                system=_STEP_SYSTEM_PROMPT,
                user=user,
                tools=tools,
                temperature=0.2,
                timeout=900.0,
                max_iterations=30,
            )
            logger.info("Step %d: done. Summary: %.200s", step_idx, response)
        except Exception as exc:
            logger.error("Step %d: FAILED: %s", step_idx, exc)
            raise RuntimeError(f"[Implementer] LLM failed for step {step_idx}: {exc}")

    def _create_pull_request(
        self, repo_url: str, branch_name: str, ticket_id: str,
        summary: str, base_branch: str | None = None,
        fork_owner: str | None = None,
    ) -> str:
        owner, repo = GitHubAdapter.parse_repo_url(repo_url)
        if base_branch is None:
            base_branch = self._github.get_repo_default_branch(owner, repo)

        # For cross-repo (fork) PRs the head must be "fork_owner:branch".
        head = f"{fork_owner}:{branch_name}" if fork_owner else branch_name

        pr = self._github.create_pull_request(
            owner=owner,
            repo=repo,
            title=f"{ticket_id} {summary}",
            head=head,
            base=base_branch,
            body=(
                f"Automated implementation for **{ticket_id}**.\n\n"
                f"**Summary:** {summary}\n\n"
                f"Branch: `{branch_name}`"
            ),
            draft=True,
        )
        return pr.html_url

    # ------------------------------------------------------------------
    # Linter auto-fix
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_linter_config(repo_path: str) -> dict[str, bool]:
        """Scan the repo for linter/formatter configuration files.

        Returns a dict of tool names mapped to whether they appear to be
        configured for the project.
        """
        root = Path(repo_path)
        detected: dict[str, bool] = {
            "ruff": False,
            "isort": False,
            "black": False,
        }

        # pyproject.toml sections
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            try:
                text = pyproject.read_text()
                if "[tool.ruff" in text or "ruff" in text.split("[build-system")[0]:
                    detected["ruff"] = True
                if "[tool.isort" in text:
                    detected["isort"] = True
                if "[tool.black" in text:
                    detected["black"] = True
            except OSError:
                pass

        # Standalone config files
        if (root / "ruff.toml").is_file() or (root / ".ruff.toml").is_file():
            detected["ruff"] = True
        if (root / ".isort.cfg").is_file():
            detected["isort"] = True

        # setup.cfg can carry isort / flake8 config
        setup_cfg = root / "setup.cfg"
        if setup_cfg.is_file():
            try:
                text = setup_cfg.read_text()
                if "[isort]" in text:
                    detected["isort"] = True
            except OSError:
                pass

        # .pre-commit-config.yaml
        precommit = root / ".pre-commit-config.yaml"
        if precommit.is_file():
            try:
                text = precommit.read_text()
                for tool in detected:
                    if tool in text:
                        detected[tool] = True
            except OSError:
                pass

        return detected

    @staticmethod
    def _try_run(cmd: list[str], cwd: str, label: str) -> bool:
        """Run *cmd* and return True if it succeeded."""
        try:
            result = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                logger.info("lint %s: OK", label)
            else:
                logger.warning(
                    "lint %s: exited %d — %s",
                    label, result.returncode, result.stderr[:300],
                )
            return result.returncode == 0
        except FileNotFoundError:
            logger.info("lint %s: not installed, skipping", label)
            return False
        except subprocess.TimeoutExpired:
            logger.warning("lint %s: timed out, skipping", label)
            return False

    def _auto_lint(self, repo_path: str) -> None:
        """Best-effort auto-format pass over the repo.

        Detects which formatters/linters the repo uses and runs their
        auto-fix modes.  Silently skips tools that are not installed.
        """
        logger.info("Running auto-lint pass")
        detected = self._detect_linter_config(repo_path)
        logger.info("lint detected config: %s", detected)

        ran_any = False

        # --- ruff (format + fix) ---
        if detected["ruff"] or shutil.which("ruff"):
            ran_any |= self._try_run(
                ["ruff", "check", "--fix", "--exit-zero", "."],
                repo_path, "ruff check --fix",
            )
            ran_any |= self._try_run(
                ["ruff", "format", "."],
                repo_path, "ruff format",
            )

        # --- isort (only if configured AND ruff doesn't handle it) ---
        ruff_handles_isort = False
        if detected["ruff"]:
            pyproject = Path(repo_path) / "pyproject.toml"
            if pyproject.is_file():
                try:
                    text = pyproject.read_text()
                    if "select" in text and "I" in text:
                        ruff_handles_isort = True
                except OSError:
                    pass

        if detected["isort"] and not ruff_handles_isort:
            ran_any |= self._try_run(
                ["isort", "."],
                repo_path, "isort",
            )

        # --- black (only if configured AND ruff format didn't run) ---
        if detected["black"] and not detected["ruff"]:
            ran_any |= self._try_run(
                ["black", "."],
                repo_path, "black",
            )

        if ran_any:
            logger.info("Auto-lint pass complete")
        else:
            logger.info("No formatters were available or configured")

    # ------------------------------------------------------------------
    # Commit and push
    # ------------------------------------------------------------------

    def _commit_and_push(
        self, commit_msg: str, branch_name: str, repo_url: str,
        fork_owner: str | None = None,
    ) -> str:
        # When using a fork, push to the fork URL rather than upstream.
        if fork_owner:
            _, repo_name = GitHubAdapter.parse_repo_url(repo_url)
            push_url = f"https://github.com/{fork_owner}/{repo_name}"
        else:
            push_url = repo_url

        return self._github.commit_and_push(
            repo_path=self.repo_path,
            repo_url=push_url,
            branch_name=branch_name,
            commit_msg=commit_msg,
        )
