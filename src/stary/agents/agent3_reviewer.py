"""Agent 3: Reviewer – acts as a senior architect performing an in-depth
code review on a GitHub pull request.  Uses the GitHub API to fetch the PR
diff and repository context (markdown docs, source files) without cloning
the repo locally.  Sends everything to an LLM for structured review, then
posts the review back to the PR via the GitHub API (APPROVE or
REQUEST_CHANGES with inline comments)."""

import base64
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

import requests

INFERENCE_URL = os.environ.get(
    "AGENT3_INFERENCE_URL",
    os.environ.get("INFERENCE_URL", "http://localhost:8080/v1/chat/completions"),
)

# File extensions considered relevant source code for context
_SOURCE_EXTENSIONS = (
    ".py", ".ts", ".js", ".yaml", ".yml", ".json", ".toml", ".cfg",
    ".ini", ".sh", ".bash", ".dockerfile",
)

# Files given higher priority when gathering context
_PRIORITY_NAMES = (
    "__init__.py", "setup.py", "setup.cfg", "pyproject.toml", "readme.md",
)

# Directories to skip when walking the repo tree from the API
_IGNORED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs",
}

_CONTEXT_MD_PATTERN = re.compile(r"\.md$", re.IGNORECASE)

# Regex to parse a GitHub PR URL
_PR_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
)


class ReviewerAgent:
    """Senior-architect code reviewer using the GitHub API.

    Pipeline
    --------
    1. Parse the PR URL to extract owner, repo, and PR number.
    2. Fetch the unified diff via the GitHub API.
    3. Fetch the repo file tree (default branch) via the GitHub API.
    4. Read markdown context docs and key source files via the API.
    5. Send everything to the LLM for review.
    6. Post the review back to the PR (APPROVE / REQUEST_CHANGES).
    """

    def __init__(
        self,
        inference_url: str | None = None,
        github_token: str | None = None,
    ):
        self.inference_url = inference_url or INFERENCE_URL
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        if not self.github_token:
            raise ValueError(
                "GITHUB_TOKEN must be set (env var or constructor arg)."
            )
        self._gh_headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self, pr_url: str) -> dict:
        """
        End-to-end review pipeline.

        Parameters
        ----------
        pr_url : str
            Full GitHub pull request URL, e.g.
            ``https://github.com/owner/repo/pull/123``

        Returns
        -------
        dict
            {
                "approved": bool,
                "comments": [...],
                "summary": str,
                "stats": {"additions": int, "deletions": int, "files": int},
                "github_review_url": str | None,
            }
        """
        # 0. Parse PR URL --------------------------------------------------
        owner, repo, pr_number = self._parse_pr_url(pr_url)
        print(f"[Agent3] Reviewing PR #{pr_number} on {owner}/{repo}")

        # 1. Fetch unified diff --------------------------------------------
        diff = self._fetch_pr_diff(owner, repo, pr_number)
        print(f"[Agent3] Fetched diff ({len(diff)} chars).")

        # 2. Fetch repo tree (default branch) ------------------------------
        repo_tree = self._fetch_repo_tree(owner, repo)
        print(f"[Agent3] Repo tree – {len(repo_tree)} file(s) found.")

        # 3. Read markdown context docs ------------------------------------
        context_docs = self._fetch_context_markdowns(owner, repo, repo_tree)
        print(f"[Agent3] Found {len(context_docs)} markdown context doc(s).")

        # 4. Read key source files -----------------------------------------
        source_snippets = self._fetch_key_sources(owner, repo, repo_tree)

        # 5. Compute basic diff stats --------------------------------------
        stats = self._compute_diff_stats(diff)
        print(
            f"[Agent3] Diff stats – {stats['additions']} addition(s), "
            f"{stats['deletions']} deletion(s), {stats['files']} file(s)."
        )

        # 6. Call LLM for the actual review --------------------------------
        review = self._review_via_llm(
            diff, repo_tree, context_docs, source_snippets,
        )

        # Ensure top-level keys are always present -------------------------
        review.setdefault("approved", False)
        review.setdefault("comments", [])
        review.setdefault("summary", "")
        review["stats"] = stats

        status = "APPROVED" if review["approved"] else (
            f"CHANGES REQUESTED ({len(review['comments'])} comment(s))"
        )
        print(f"[Agent3] Review verdict: {status}")
        print(f"[Agent3] Summary: {review['summary']}")
        for i, c in enumerate(review["comments"], 1):
            sev = c.get("severity", "info").upper()
            fpath = c.get("file", "general")
            print(f"[Agent3]   #{i} [{sev}] {fpath}: {c.get('comment', '')}")

        # 7. Post review comments to GitHub --------------------------------
        comment_url = self._post_pr_comment(
            owner, repo, pr_number, review,
        )
        review["github_comment_url"] = comment_url

        # 8. Rebase & merge if approved ------------------------------------
        review["merged"] = False
        if review["approved"]:
            merged = self._merge_pr(owner, repo, pr_number)
            review["merged"] = merged

        print(f"[Agent3] Review complete: {status}")
        return review

    # ------------------------------------------------------------------
    # 0. Parse PR URL
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pr_url(pr_url: str) -> tuple[str, str, int]:
        """Extract (owner, repo, pr_number) from a GitHub PR URL."""
        m = _PR_URL_RE.search(pr_url)
        if not m:
            raise ValueError(
                f"Invalid GitHub PR URL: {pr_url!r}.  "
                "Expected format: https://github.com/OWNER/REPO/pull/NUMBER"
            )
        return m.group("owner"), m.group("repo"), int(m.group("number"))

    # ------------------------------------------------------------------
    # 1. Fetch PR diff via GitHub API
    # ------------------------------------------------------------------

    def _fetch_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Download the unified diff for the pull request."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        headers = {
            **self._gh_headers,
            "Accept": "application/vnd.github.v3.diff",
        }
        print(f"[Agent3] GET {url} (diff)")
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.text

    # ------------------------------------------------------------------
    # 2. Fetch repo file tree (default branch, recursive)
    # ------------------------------------------------------------------

    def _fetch_repo_tree(self, owner: str, repo: str) -> list[str]:
        """Return a sorted list of repo-relative file paths via the Git
        Trees API (recursive)."""
        # First, get the default branch SHA
        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        resp = requests.get(repo_url, headers=self._gh_headers, timeout=30)
        resp.raise_for_status()
        default_branch = resp.json().get("default_branch", "main")

        tree_url = (
            f"https://api.github.com/repos/{owner}/{repo}"
            f"/git/trees/{default_branch}?recursive=1"
        )
        print(f"[Agent3] GET {tree_url}")
        resp = requests.get(tree_url, headers=self._gh_headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        paths: list[str] = []
        for item in data.get("tree", []):
            if item.get("type") != "blob":
                continue
            path = item["path"]
            # skip ignored directories
            parts = path.split("/")
            if any(p in _IGNORED_DIRS for p in parts):
                continue
            paths.append(path)
        return sorted(paths)

    # ------------------------------------------------------------------
    # 3. Fetch markdown context files via GitHub API
    # ------------------------------------------------------------------

    def _fetch_context_markdowns(
        self,
        owner: str,
        repo: str,
        repo_tree: list[str],
    ) -> dict[str, str]:
        """Fetch every .md file from the repo as architecture context."""
        docs: dict[str, str] = {}
        for rel in repo_tree:
            if _CONTEXT_MD_PATTERN.search(rel):
                content = self._fetch_file_content(owner, repo, rel)
                if content is not None:
                    docs[rel] = content
        return docs

    # ------------------------------------------------------------------
    # 4. Fetch key source files
    # ------------------------------------------------------------------

    def _fetch_key_sources(
        self,
        owner: str,
        repo: str,
        repo_tree: list[str],
        max_files: int = 30,
        max_bytes: int = 80_000,
    ) -> dict[str, str]:
        """Fetch relevant source files so the LLM has real code context."""
        priority: list[str] = []
        rest: list[str] = []

        for rel in repo_tree:
            lower = rel.lower()
            basename = lower.rsplit("/", 1)[-1]
            if basename in _PRIORITY_NAMES:
                priority.append(rel)
            elif lower.endswith(_SOURCE_EXTENSIONS):
                rest.append(rel)

        candidates = priority + rest
        snippets: dict[str, str] = {}
        total = 0
        for rel in candidates[:max_files]:
            text = self._fetch_file_content(owner, repo, rel)
            if text is None:
                continue
            if total + len(text) > max_bytes:
                text = text[: max_bytes - total] + "\n... (truncated)"
            snippets[rel] = text
            total += len(text)
            if total >= max_bytes:
                break

        print(f"[Agent3] Fetched {len(snippets)} source file(s) ({total} bytes).")
        return snippets

    # ------------------------------------------------------------------
    # GitHub file content helper
    # ------------------------------------------------------------------

    def _fetch_file_content(
        self, owner: str, repo: str, path: str
    ) -> str | None:
        """Fetch a single file's contents from the repo (default branch)."""
        url = (
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        )
        try:
            resp = requests.get(url, headers=self._gh_headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("encoding") == "base64":
                return base64.b64decode(data["content"]).decode(
                    "utf-8", errors="replace"
                )
            # Fallback: download_url
            dl = data.get("download_url")
            if dl:
                return requests.get(dl, timeout=30).text
        except Exception as exc:
            print(f"[Agent3] Could not fetch {path}: {exc}")
        return None

    # ------------------------------------------------------------------
    # 5. Diff stats (computed locally, not by the LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_diff_stats(diff: str) -> dict:
        additions = sum(
            1 for line in diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        deletions = sum(
            1 for line in diff.splitlines()
            if line.startswith("-") and not line.startswith("---")
        )
        files = diff.count("+++ b/") + diff.count("+++ /dev/null")
        files = max(files, diff.count("--- a/") + diff.count("--- /dev/null"))
        return {"additions": additions, "deletions": deletions, "files": files}

    # ------------------------------------------------------------------
    # 6. LLM-powered review
    # ------------------------------------------------------------------

    def _review_via_llm(
        self,
        diff: str,
        repo_tree: list[str],
        context_docs: dict[str, str],
        source_snippets: dict[str, str],
    ) -> dict:
        """Send the diff together with repo context to the LLM for review."""

        tree_str = "\n".join(repo_tree)
        context_block = "\n\n".join(
            f"### {path}\n{content}" for path, content in context_docs.items()
        )
        source_block = "\n\n".join(
            f"### {path}\n```\n{content}\n```"
            for path, content in source_snippets.items()
        )

        system = (
            "You are a senior software architect performing a thorough code "
            "review on a unified diff.\n\n"
            "You are provided with:\n"
            "1. The unified diff to review.\n"
            "2. The repository file tree.\n"
            "3. Architecture / convention markdown documents from the repo "
            "(these are the source of truth for naming, structure, patterns).\n"
            "4. Key existing source files from the repository.\n\n"
            "Your responsibilities:\n"
            "- **Architecture alignment**: verify the diff does not break the "
            "established project architecture or violate conventions defined "
            "in the markdown documents.\n"
            "- **Correctness**: look for logic errors, off-by-one mistakes, "
            "wrong return types, missing error handling, race conditions, or "
            "any code that would malfunction at runtime.\n"
            "- **Security**: flag uses of eval(), exec(), unsafe deserialization, "
            "SQL injection vectors, or secrets in code.\n"
            "- **Code quality**: identify TODO/FIXME placeholders, "
            "NotImplementedError stubs, dead code, missing docstrings on "
            "public APIs, or violations of project style.\n"
            "- **Completeness**: check that all required changes are present "
            "(e.g. exports, imports, test coverage) and nothing is left "
            "half-implemented.\n\n"
            "For EVERY concern, produce a comment with a severity level.\n\n"
            "Finally, decide whether this diff is safe to merge.\n\n"
            "Return ONLY valid JSON (no markdown fences) with this schema:\n"
            "{\n"
            '  "approved": true/false,\n'
            '  "summary": "<one-paragraph overall assessment>",\n'
            '  "comments": [\n'
            "    {\n"
            '      "severity": "critical" | "warning" | "suggestion",\n'
            '      "file": "<affected file path or \'general\'>",\n'
            '      "comment": "<detailed description of the concern>"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- `approved` must be FALSE if there are any `critical` comments.\n"
            "- `approved` may be TRUE if there are only `warning` or "
            "`suggestion` comments and none are blocking.\n"
            "- Be specific: reference file names, function names, and line "
            "context from the diff.\n"
            "- Do NOT invent issues that are not evidenced in the diff or "
            "repo context.\n"
        )

        user = (
            f"## Unified diff to review\n```diff\n{diff}\n```\n\n"
            f"## Repository file tree\n```\n{tree_str}\n```\n\n"
            f"## Architecture & convention documents\n"
            f"{context_block or '(none found)'}\n\n"
            f"## Existing source code\n{source_block or '(none)'}\n\n"
            "Perform your review now."
        )

        return self._call_llm(system, user, tag="review")

    # ------------------------------------------------------------------
    # 7. Post review comments to GitHub
    # ------------------------------------------------------------------

    def _post_pr_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        review: dict,
    ) -> str | None:
        """Post the review as a PR comment (issue comment).

        All review feedback (verdict, summary, individual comments) is
        rendered as a single formatted comment on the pull request.
        """
        verdict = "APPROVED ✅" if review.get("approved") else "CHANGES REQUESTED ❌"

        # Build a nicely formatted comment body
        body_parts: list[str] = []
        body_parts.append("## Agent 3 – Automated Code Review\n")
        body_parts.append(f"**Verdict:** {verdict}\n")
        if review.get("summary"):
            body_parts.append(f"**Summary:** {review['summary']}\n")

        stats = review.get("stats", {})
        if stats:
            body_parts.append(
                f"**Stats:** +{stats.get('additions', 0)} "
                f"-{stats.get('deletions', 0)} "
                f"in {stats.get('files', 0)} file(s)\n"
            )

        comments = review.get("comments", [])
        if comments:
            body_parts.append("### Comments\n")
            for i, c in enumerate(comments, 1):
                sev = c.get("severity", "info")
                icon = {"critical": "🔴", "warning": "🟡", "suggestion": "🔵"}.get(
                    sev, "⚪"
                )
                fpath = c.get("file", "general")
                body_parts.append(
                    f"{i}. {icon} **[{sev.upper()}]** `{fpath}` — "
                    f"{c.get('comment', '')}\n"
                )

        if review.get("approved"):
            body_parts.append("\n---\n\n🚀 **Auto-merging via rebase.**\n")

        body = "\n".join(body_parts)

        url = (
            f"https://api.github.com/repos/{owner}/{repo}"
            f"/issues/{pr_number}/comments"
        )
        print(f"[Agent3] POST {url}  (review comment)")
        try:
            resp = requests.post(
                url, json={"body": body},
                headers=self._gh_headers, timeout=60,
            )
            resp.raise_for_status()
            comment_url = resp.json().get("html_url")
            print(f"[Agent3] PR comment posted: {comment_url}")
            return comment_url
        except requests.RequestException as exc:
            print(f"[Agent3] Failed to post PR comment: {exc}")
            if hasattr(exc, "response") and exc.response is not None:
                print(f"[Agent3] Response body: {exc.response.text}")
            return None

    # ------------------------------------------------------------------
    # 8. Merge PR via rebase
    # ------------------------------------------------------------------

    def _merge_pr(
        self, owner: str, repo: str, pr_number: int
    ) -> bool:
        """Merge the PR using the rebase strategy.

        Returns True if the merge succeeded, False otherwise.
        """
        url = (
            f"https://api.github.com/repos/{owner}/{repo}"
            f"/pulls/{pr_number}/merge"
        )
        payload = {"merge_method": "rebase"}
        print(f"[Agent3] PUT {url}  merge_method=rebase")
        try:
            resp = requests.put(
                url, json=payload, headers=self._gh_headers, timeout=60,
            )
            resp.raise_for_status()
            msg = resp.json().get("message", "")
            print(f"[Agent3] PR #{pr_number} merged successfully: {msg}")
            return True
        except requests.RequestException as exc:
            print(f"[Agent3] Failed to merge PR #{pr_number}: {exc}")
            if hasattr(exc, "response") and exc.response is not None:
                print(f"[Agent3] Response body: {exc.response.text}")
            return False

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
        """Send a chat-completion request and parse the JSON response."""
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        url = self.inference_url
        label = f"[Agent3/{tag}]" if tag else "[Agent3]"
        print(f"{label} Calling LLM at {url} …")

        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            content: str = data["choices"][0]["message"]["content"]

            content = content.strip()
            fence = re.search(r"```(?:json)?\s*\n(.*?)```", content, re.DOTALL)
            if fence:
                content = fence.group(1).strip()
            return json.loads(content)

        except requests.RequestException as exc:
            print(f"{label} LLM request failed: {exc}")
            return {"approved": False, "comments": [], "summary": f"LLM request failed: {exc}"}
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            print(f"{label} Failed to parse LLM response: {exc}")
            return {"approved": False, "comments": [], "summary": f"Failed to parse LLM response: {exc}"}
