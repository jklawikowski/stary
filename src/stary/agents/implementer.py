"""Implementer – takes validated steps from the Planner, iterates over each
step calling the LLM separately, writes results to the cloned repo, commits,
pushes the changes to a feature branch, and creates a pull request."""

import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from stary.inference import BaseInferenceClient, InferenceClient, get_inference_client

# ---------------------------------------------------------------------------
# Prompt budget constants – per-step budgets are smaller because each LLM
# call handles only one step.  Characters ≈ tokens × 4.
# ---------------------------------------------------------------------------
_MAX_PROMPT_CHARS = 200_000
_TREE_BUDGET      = 20_000   # repo file-tree listing
_CONTEXT_BUDGET   = 30_000   # markdown docs (architecture, conventions, …)
_SOURCE_BUDGET    = 60_000   # source code relevant to the current step


class Implementer:
    """Iterates over discrete implementation steps from the Planner,
    calling the LLM once per step.  After all steps are implemented the
    changes are committed, pushed, and a pull request is created."""

    _STEP_MAX_RETRIES = 3

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
        self.repo_path: str | None = None  # set from planner output

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self, planner_output: dict) -> str:
        """End-to-end implementation pipeline.

        Parameters
        ----------
        planner_output : dict
            Output from the Planner:
            - steps       : list[dict]  each has 'prompt' (str) plus
                            optional keys like 'target_files'
            - repo_url         : str
            - repo_path        : str    (local clone path)
            - branch_name      : str
            - repo_tree        : list[str]
            - context_docs     : dict[str, str]
            - ticket_id        : str
            - summary          : str

        Returns
        -------
        str  – URL to the created pull request.
        """
        repo_url = planner_output["repo_url"]
        self.repo_path = planner_output["repo_path"]
        branch_name = planner_output["branch_name"]
        repo_tree = planner_output["repo_tree"]
        context_docs = planner_output["context_docs"]
        ticket_id = planner_output["ticket_id"]
        summary = planner_output["summary"]
        steps = planner_output["steps"]

        print(f"[Implementer] Ticket   : {ticket_id}")
        print(f"[Implementer] Repo     : {repo_url}")
        print(f"[Implementer] Branch   : {branch_name}")
        print(f"[Implementer] Steps    : {len(steps)}")

        # Pre-compute shared context strings (built once, reused per step)
        tree_str = self._build_tree_string(repo_tree)
        context_block = self._cap_context_block(context_docs, _CONTEXT_BUDGET)

        # 1. Iterate over each step, calling LLM separately ----------------
        total_ops = 0
        for idx, step in enumerate(steps, 1):
            label = step.get("prompt", "")[:60]
            print(f"\n[Implementer] --- Step {idx}/{len(steps)}: {label}... ---")

            # Read source files relevant to THIS step
            step_sources = self._read_step_sources(step, repo_tree)

            # Call LLM for this single step
            file_ops = self._implement_step(
                step, idx, len(steps),
                tree_str, context_block, step_sources,
            )
            print(f"[Implementer]   LLM produced {len(file_ops)} file op(s)")

            # Write files immediately so subsequent steps see the changes
            self._write_files(file_ops)
            total_ops += len(file_ops)

        print(f"\n[Implementer] All steps done — {total_ops} file op(s) total.")

        # 2. Commit & push -------------------------------------------------
        commit_msg = f"{ticket_id} {summary}"
        branch_url = self._commit_and_push(commit_msg, branch_name, repo_url)
        print(f"[Implementer] Pushed to: {branch_url}")

        # 3. Create pull request -------------------------------------------
        pr_url = self._create_pull_request(
            repo_url, branch_name, ticket_id, summary,
        )
        print(f"[Implementer] PR created: {pr_url}")

        return pr_url

    # ------------------------------------------------------------------
    # Read sources relevant to a single step
    # ------------------------------------------------------------------

    def _read_step_sources(
        self,
        step: dict,
        repo_tree: list[str],
        budget: int = _SOURCE_BUDGET,
    ) -> dict[str, str]:
        """Read source files relevant to *step* from the cloned repo.

        Includes:
        1. Files listed in ``target_files``.
        2. Sibling files in the same directories (for import context).
        3. ``__init__.py`` along the path of target files.
        """
        target_paths: set[str] = set(step.get("target_files", []))
        target_dirs: set[str] = {os.path.dirname(p) for p in target_paths}

        # Build candidate list: targets first, then siblings, then __init__
        candidates: list[str] = []
        siblings: list[str] = []
        inits: list[str] = []

        for rel in repo_tree:
            if rel in target_paths:
                candidates.append(rel)
            elif os.path.dirname(rel) in target_dirs:
                siblings.append(rel)
            elif rel.endswith("__init__.py"):
                pkg_dir = os.path.dirname(rel)
                if any(td.startswith(pkg_dir) for td in target_dirs):
                    inits.append(rel)

        ordered = candidates + siblings + inits

        result: dict[str, str] = {}
        total = 0
        for rel in ordered:
            full = os.path.join(self.repo_path, rel)
            try:
                text = Path(full).read_text(errors="replace")
            except OSError:
                continue
            if total + len(text) > budget:
                remaining = budget - total
                if remaining > 200:
                    result[rel] = text[:remaining] + "\n... (truncated)"
                    total = budget
                break
            result[rel] = text
            total += len(text)

        print(f"[Implementer]   Read {len(result)} source file(s) ({total} bytes) for step")
        return result

    # ------------------------------------------------------------------
    # Implement a single step (with retry)
    # ------------------------------------------------------------------

    def _implement_step(
        self,
        step: dict,
        step_idx: int,
        total_steps: int,
        tree_str: str,
        context_block: str,
        step_sources: dict[str, str],
    ) -> list[dict]:
        """Call the LLM to implement one step, with retry on failure."""

        system = self._step_system_prompt()
        user = self._build_step_user(
            step, step_idx, total_steps,
            tree_str, context_block, step_sources,
        )

        for attempt in range(1, self._STEP_MAX_RETRIES + 1):
            label = f"step{step_idx}/attempt-{attempt}"
            ops = self._try_implement(system, user, attempt_label=label)
            if ops is not None:
                return ops
            print(f"[Implementer/{label}] failed, retrying...")

        raise RuntimeError(
            f"[Implementer] LLM failed to produce valid file operations "
            f"for step {step_idx} "
            f"after {self._STEP_MAX_RETRIES} attempts."
        )

    # ------------------------------------------------------------------
    # Step prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _step_system_prompt() -> str:
        return (
            "You are an expert software engineer. Your job is to implement "
            "ONE specific step by producing the FULL content of every "
            "file that needs to be created or modified for this step.\n\n"
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
            "6. Implement the step fully — do not leave TODOs or placeholders.\n\n"
            "Return ONLY a JSON array (no markdown fences) with this schema:\n"
            "[\n"
            '  {"path": "<repo-relative path>", '
            '"action": "create|modify|delete", '
            '"content": "<full file content>"},\n'
            "  ...\n"
            "]\n"
        )

    def _build_step_user(
        self,
        step: dict,
        step_idx: int,
        total_steps: int,
        tree_str: str,
        context_block: str,
        step_sources: dict[str, str],
    ) -> str:
        """Build the user prompt for a single implementation step.

        Uses the step's ``prompt`` field as the core instruction and
        supplements it with repo tree, context docs, and source code.
        Any additional keys in *step* (beyond ``prompt`` and
        ``target_files``) are serialised as extra context.
        """
        step_prompt = step.get("prompt", "")

        # Collect any extra metadata the Planner attached to the step
        extra_keys = {
            k: v for k, v in step.items()
            if k not in ("prompt", "target_files") and v
        }
        extra_block = ""
        if extra_keys:
            extra_block = (
                "## Additional step metadata\n"
                f"```json\n{json.dumps(extra_keys, indent=2)}\n```\n\n"
            )

        source_block = "\n\n".join(
            f"### {path}\n```\n{content}\n```"
            for path, content in step_sources.items()
        )

        user = (
            f"## Implementation instruction (step {step_idx} of {total_steps})\n"
            f"{step_prompt}\n\n"
            f"{extra_block}"
            f"## Repository file tree\n```\n{tree_str}\n```\n\n"
            f"## Architecture / convention documents\n{context_block or '(none)'}\n\n"
            f"## Existing source code (files relevant to this step)\n"
            f"{source_block or '(none)'}\n\n"
            "Generate the JSON array of file operations for THIS step only."
        )

        prompt_size = len(self._step_system_prompt()) + len(user)
        if prompt_size > _MAX_PROMPT_CHARS:
            overshoot = prompt_size - _MAX_PROMPT_CHARS
            print(
                f"[Implementer/step{step_idx}] WARNING: prompt ({prompt_size} chars) "
                f"exceeds budget. Trimming source block by {overshoot} chars."
            )
            max_source = len(source_block) - overshoot - 200
            if max_source > 200:
                source_block_trimmed = source_block[:max_source] + "\n... (trimmed)"
            else:
                source_block_trimmed = "(trimmed — step instruction and tree should suffice)"
            user = (
                f"## Implementation instruction (step {step_idx} of {total_steps})\n"
                f"{step_prompt}\n\n"
                f"{extra_block}"
                f"## Repository file tree\n```\n{tree_str}\n```\n\n"
                f"## Architecture / convention documents\n{context_block or '(none)'}\n\n"
                f"## Existing source code (files relevant to this step)\n"
                f"{source_block_trimmed}\n\n"
                "Generate the JSON array of file operations for THIS step only."
            )

        print(f"[Implementer/step{step_idx}] Prompt size: {prompt_size} chars")
        return user

    # ------------------------------------------------------------------
    # Single LLM attempt (shared with retry logic)
    # ------------------------------------------------------------------

    def _try_implement(
        self,
        system: str,
        user: str,
        *,
        attempt_label: str = "",
    ) -> list[dict] | None:
        """Single LLM call for implementation.  Returns a list of file ops
        on success, or ``None`` on failure.  Uses partial-array recovery
        to salvage truncated responses."""
        tag = f"implement/{attempt_label}" if attempt_label else "implement"

        try:
            raw_text = self._call_llm(
                system, user, tag=tag, raw=True, propagate_errors=True,
            )
        except Exception as exc:
            print(f"[Implementer/{tag}] Inference error: {exc}")
            return None

        if not raw_text or len(raw_text) < 3:
            print(f"[Implementer/{tag}] Empty or trivial response ({len(raw_text) if raw_text else 0} chars)")
            return None

        print(f"[Implementer/{tag}] Raw output ({len(raw_text)} chars):\n{raw_text[:500]}...")

        # Try full parse first
        parsed = BaseInferenceClient.extract_json_array(raw_text)
        if isinstance(parsed, list) and parsed:
            print(f"[Implementer/{tag}] Full parse: {len(parsed)} file op(s)")
            return parsed
        if isinstance(parsed, dict):
            for key in ("files", "file_ops", "operations", "changes"):
                if isinstance(parsed.get(key), list) and parsed[key]:
                    print(f"[Implementer/{tag}] Found {len(parsed[key])} op(s) under key '{key}'")
                    return parsed[key]

        # Try partial recovery (truncated array)
        partial = BaseInferenceClient.extract_partial_json_array(raw_text)
        if partial:
            print(
                f"[Implementer/{tag}] Partial recovery: {len(partial)} complete "
                f"file op(s) from truncated response"
            )
            return partial

        print(f"[Implementer/{tag}] Could not parse any file ops from response")
        return None

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
                "[Implementer] github_token is required to create a pull request. "
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
                f"[Implementer] Cannot extract owner/repo from URL: {repo_url}"
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
                    print(f"[Implementer]   deleted {rel}")
                continue

            # create / modify → write content
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(op.get("content", ""))
            print(f"[Implementer]   wrote   {rel}")

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
    # Prompt-building helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tree_string(
        repo_tree: list[str],
        budget: int = _TREE_BUDGET,
    ) -> str:
        """Build a repo tree string that fits within *budget* characters."""
        full = "\n".join(repo_tree)
        if len(full) <= budget:
            return full

        # Fall back to directory summary
        dir_counts: Counter[str] = Counter()
        for p in repo_tree:
            parts = p.split("/")
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

    @staticmethod
    def _cap_context_block(
        context_docs: dict[str, str], budget: int
    ) -> str:
        """Build a context-docs string capped to *budget* characters."""
        parts: list[str] = []
        total = 0
        for path, content in context_docs.items():
            block = f"### {path}\n{content}"
            if total + len(block) > budget:
                remaining = budget - total
                if remaining > 200:
                    parts.append(block[:remaining] + "\n... (truncated)")
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)

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
        propagate_errors: bool = False,
    ) -> Any:
        """Send a chat-completion request via inference client.

        Parameters
        ----------
        raw : bool
            If True, return the raw text content instead of parsing as JSON.
        propagate_errors : bool
            If True, re-raise exceptions instead of returning a fallback
            value.  Useful when the caller needs to distinguish inference
            failures from empty LLM responses.
        """
        label = f"[Implementer/{tag}]" if tag else "[Implementer]"
        print(f"{label} Calling inference…")

        try:
            if raw:
                content = self._inference.chat(
                    system=system,
                    user=user,
                    temperature=temperature,
                    timeout=900.0,
                )
                print(f"{label} Got raw response ({len(content)} chars)")
                return content.strip() if content else ""
            else:
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
            if propagate_errors:
                raise
            return "" if raw else {}
