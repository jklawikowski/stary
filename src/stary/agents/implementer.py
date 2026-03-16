"""Implementer -- gives the LLM tools to read and write files directly,
then commits, pushes, and creates a pull request.

One tool-calling session is run per implementation step.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from stary.agents.tools import make_read_tools, make_shell_tools, make_write_tools
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
        github_token: str | None = None,
        git_user_name: str | None = None,
        git_user_email: str | None = None,
    ):
        self._inference = inference_client or get_inference_client()
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        self.git_user_name = git_user_name or os.environ.get("GIT_USER_NAME", "qaplatformbot")
        self.git_user_email = git_user_email or os.environ.get("GIT_USER_EMAIL", "sys_qaplatformbot@intel.com")
        self.repo_path: str | None = None

    def run(self, planner_output: dict) -> str:
        """End-to-end implementation pipeline.  Returns PR URL."""
        repo_url = planner_output["repo_url"]
        self.repo_path = planner_output["repo_path"]
        branch_name = planner_output["branch_name"]
        ticket_id = planner_output["ticket_id"]
        summary = planner_output["summary"]
        steps = planner_output["steps"]

        print(f"[Implementer] Ticket   : {ticket_id}")
        print(f"[Implementer] Repo     : {repo_url}")
        print(f"[Implementer] Branch   : {branch_name}")
        print(f"[Implementer] Steps    : {len(steps)}")

        # Build tools for this repo
        read_tools = make_read_tools(self.repo_path)
        write_tools = make_write_tools(self.repo_path)
        shell_tools = make_shell_tools(self.repo_path)
        all_tools = read_tools + write_tools + shell_tools

        for idx, step in enumerate(steps, 1):
            label = step.get("prompt", "")[:60]
            print(f"\n[Implementer] --- Step {idx}/{len(steps)}: {label}... ---")
            self._implement_step(step, idx, len(steps), all_tools)

        print(f"\n[Implementer] All {len(steps)} step(s) done.")

        self._auto_lint(self.repo_path)

        commit_msg = f"{ticket_id} {summary}"
        branch_url = self._commit_and_push(commit_msg, branch_name, repo_url)
        print(f"[Implementer] Pushed to: {branch_url}")

        pr_url = self._create_pull_request(repo_url, branch_name, ticket_id, summary)
        print(f"[Implementer] PR created: {pr_url}")
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

        print(f"[Implementer/step{step_idx}] Starting tool-calling session...")
        try:
            response = self._inference.chat_with_tools(
                system=_STEP_SYSTEM_PROMPT,
                user=user,
                tools=tools,
                temperature=0.2,
                timeout=900.0,
                max_iterations=30,
            )
            print(f"[Implementer/step{step_idx}] Done. Summary: {response[:200]}")
        except Exception as exc:
            print(f"[Implementer/step{step_idx}] FAILED: {exc}")
            raise RuntimeError(f"[Implementer] LLM failed for step {step_idx}: {exc}")

    def _create_pull_request(
        self, repo_url: str, branch_name: str, ticket_id: str,
        summary: str, base_branch: str = "master",
    ) -> str:
        if not self.github_token:
            raise EnvironmentError("[Implementer] github_token is required to create a PR.")

        url_clean = repo_url.rstrip("/")
        if url_clean.endswith(".git"):
            url_clean = url_clean[:-4]
        parsed = urlparse(url_clean)
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"[Implementer] Cannot extract owner/repo from: {repo_url}")
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

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                resp = requests.post(api_url, json=body, headers=headers, timeout=60)
                resp.raise_for_status()
                return resp.json()["html_url"]
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                print(f"[Implementer] PR creation attempt {attempt}/3 failed: {exc}")
                if attempt < 3:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"[Implementer] Failed to create PR after 3 attempts: {last_exc}")

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
                print(f"[Implementer/lint] {label}: OK")
            else:
                print(
                    f"[Implementer/lint] {label}: exited {result.returncode}"
                    f" — {result.stderr[:300]}"
                )
            return result.returncode == 0
        except FileNotFoundError:
            print(f"[Implementer/lint] {label}: not installed, skipping")
            return False
        except subprocess.TimeoutExpired:
            print(f"[Implementer/lint] {label}: timed out, skipping")
            return False

    def _auto_lint(self, repo_path: str) -> None:
        """Best-effort auto-format pass over the repo.

        Detects which formatters/linters the repo uses and runs their
        auto-fix modes.  Silently skips tools that are not installed.
        """
        print("[Implementer] Running auto-lint pass...")
        detected = self._detect_linter_config(repo_path)
        print(f"[Implementer/lint] Detected config: {detected}")

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
            print("[Implementer] Auto-lint pass complete.")
        else:
            print("[Implementer] No formatters were available or configured.")

    # ------------------------------------------------------------------
    # Commit and push
    # ------------------------------------------------------------------

    def _commit_and_push(self, commit_msg: str, branch_name: str, repo_url: str) -> str:
        run = lambda cmd: subprocess.run(
            cmd, cwd=self.repo_path, check=True, capture_output=True, text=True,
        )

        run(["git", "config", "user.name", self.git_user_name])
        run(["git", "config", "user.email", self.git_user_email])
        run(["git", "add", "-A"])

        # Check if there are staged changes before committing
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=self.repo_path, capture_output=True,
        )
        if status.returncode == 0:
            raise RuntimeError(
                "[Implementer] No files were changed by the implementation steps. "
                "Nothing to commit."
            )

        run(["git", "commit", "-m", commit_msg])

        if self.github_token and "github.com" in repo_url:
            auth_url = repo_url.replace(
                "https://github.com", f"https://{self.github_token}@github.com",
            )
            run(["git", "push", auth_url, branch_name])
        else:
            run(["git", "push", "origin", branch_name])

        base = repo_url.rstrip("/")
        if base.endswith(".git"):
            base = base[:-4]
        return f"{base}/tree/{branch_name}"
