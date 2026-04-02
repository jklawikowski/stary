"""Reviewer -- senior architect code review on a GitHub pull request,
powered by tool-calling.

The LLM uses tools to fetch the PR diff, explore repo files, and read
context. It produces a structured review which is then posted back
to the PR via the GitHub API.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

from stary.agents.tools import make_github_review_tools
from stary.github_adapter import GitHubAdapter
from stary.inference import InferenceClient, get_inference_client

_REVIEW_SYSTEM_PROMPT = """\
You are a senior software architect performing a thorough code review
on a GitHub pull request.

You have tools to:
- get_pr_diff: Get the unified diff for the PR being reviewed.
- get_pr_info: Get PR metadata (title, author, body, stats).
- get_pr_changed_files: List all changed files with their status.
- list_repo_files: List all files in the repository.
- read_repo_file: Read a file from the repository (default branch).

Your workflow:
1. Start by getting the PR diff and PR info.
2. Examine the changed files list.
3. Read relevant existing files from the repo for context (architecture
   docs, related source files, tests).
4. Produce your review.

Your responsibilities:
- Architecture alignment: verify changes follow project conventions.
- Correctness: look for logic errors, wrong types, missing error handling.
- Security: flag eval(), exec(), unsafe deserialization, injection vectors.
- Code quality: identify TODO/FIXME stubs, dead code, style violations.
- Completeness: check all required changes are present (imports, tests, etc.).

When you are done reviewing, provide your FINAL answer as ONLY valid JSON
(no markdown fences) with this schema:
{
  "approved": true/false,
  "summary": "<one-paragraph overall assessment>",
  "comments": [
    {
      "severity": "critical" | "warning" | "suggestion",
      "file": "<affected file path or 'general'>",
      "comment": "<detailed description>"
    }
  ]
}

Rules:
- approved must be FALSE if there are any critical comments.
- approved may be TRUE with only warning/suggestion comments.
- Be specific: reference file names, function names, and line context.
- Do NOT invent issues not evidenced in the diff or repo.
"""


class Reviewer:
    """Senior-architect code reviewer using tool-calling and the GitHub API."""

    def __init__(
        self,
        inference_client: InferenceClient | None = None,
        github: GitHubAdapter | None = None,
    ):
        self._inference = inference_client or get_inference_client()
        self._github = github or GitHubAdapter()
        if not self._github.token:
            raise ValueError("GITHUB_TOKEN must be set (env var or constructor arg).")

    def run(self, pr_url: str, auto_merge: bool = True) -> dict:
        """End-to-end review pipeline.

        Returns dict with approved, comments, summary, stats, merged.
        """
        owner, repo, pr_number = GitHubAdapter.parse_pr_url(pr_url)
        logger.info("Reviewing PR #%d on %s/%s", pr_number, owner, repo)

        # Build tools for this PR
        tools = make_github_review_tools(
            self._github, owner, repo, pr_number,
        )

        # Run LLM with tools
        logger.info("Starting tool-calling review session")
        try:
            review = self._inference.chat_json_with_tools(
                system=_REVIEW_SYSTEM_PROMPT,
                user=(
                    f"Review pull request #{pr_number} on {owner}/{repo}.\n"
                    f"PR URL: {pr_url}\n\n"
                    "Start by fetching the PR diff and info, then explore "
                    "the repo for context before producing your review."
                ),
                tools=tools,
                temperature=0.2,
                timeout=1800.0,
                max_iterations=20,
            )
        except Exception as exc:
            logger.error("LLM failed: %s", exc)
            review = {}

        # Ensure defaults
        review.setdefault("approved", False)
        review.setdefault("comments", [])
        review.setdefault("summary", "")

        # Compute diff stats
        stats = self._compute_diff_stats(owner, repo, pr_number)
        review["stats"] = stats

        status = "APPROVED" if review["approved"] else (
            f"CHANGES REQUESTED ({len(review['comments'])} comment(s))"
        )
        logger.info("Verdict: %s", status)
        logger.info("Summary: %s", review["summary"])

        # Post review to GitHub
        comment_url = self._post_pr_comment(owner, repo, pr_number, review)
        review["github_comment_url"] = comment_url

        # Mark as ready for review if approved (PR is created as draft)
        review["merged"] = False
        if review["approved"]:
            ready = self._github.mark_pr_ready_for_review(owner, repo, pr_number)
            if not ready:
                logger.warning("Failed to mark PR #%d as ready for review", pr_number)
            if auto_merge:
                review["merged"] = self._github.merge_pull_request(owner, repo, pr_number)
            else:
                logger.info("PR approved but auto_merge=False, skipping merge")
        else:
            logger.info("PR #%d not approved, keeping as draft", pr_number)

        return review

    def _compute_diff_stats(self, owner: str, repo: str, pr_number: int) -> dict:
        try:
            pr = self._github.get_pull_request(owner, repo, pr_number)
            return {
                "additions": pr.additions,
                "deletions": pr.deletions,
                "files": pr.changed_files,
            }
        except Exception:
            return {"additions": 0, "deletions": 0, "files": 0}

    def _post_pr_comment(self, owner: str, repo: str, pr_number: int, review: dict) -> str | None:
        verdict = "APPROVED \u2705" if review.get("approved") else "CHANGES REQUESTED \u274c"
        body_parts = []
        body_parts.append("## Reviewer -- Automated Code Review\n")
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
                icon = {"critical": "\U0001f534", "warning": "\U0001f7e1", "suggestion": "\U0001f535"}.get(sev, "\u26aa")
                fpath = c.get("file", "general")
                body_parts.append(
                    f"{i}. {icon} **[{sev.upper()}]** `{fpath}` -- "
                    f"{c.get('comment', '')}\n"
                )

        body = "\n".join(body_parts)
        return self._github.post_issue_comment(owner, repo, pr_number, body)

