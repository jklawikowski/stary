"""Low-level GitHub REST API and Git adapter.

Provides a clean, reusable interface for GitHub operations with:
- Centralized authentication
- Retry logic for transient failures
- Consistent error handling
- Local git operations (clone, branch, commit, push)

This module has NO business logic — it's a pure data-access layer.
"""

import base64
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from stary.telemetry import tracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GIT_USER_NAME = os.environ.get("GIT_USER_NAME", "qaplatformbot")
GIT_USER_EMAIL = os.environ.get("GIT_USER_EMAIL", "sys_qaplatformbot@intel.com")

DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5

_PR_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
)


def _normalise_github_route(endpoint: str) -> str:
    """Replace dynamic path segments with placeholders for GitHub API routes.

    >>> _normalise_github_route("/repos/owner/repo/pulls/123")
    '/repos/{owner}/{repo}/pulls/{number}'
    >>> _normalise_github_route("/repos/owner/repo/contents/src/main.py")
    '/repos/{owner}/{repo}/contents/{path}'
    >>> _normalise_github_route("/user")
    '/user'
    """
    import re as _re

    # /repos/{owner}/{repo}/...
    result = _re.sub(
        r"^/repos/[^/]+/[^/]+",
        "/repos/{owner}/{repo}",
        endpoint,
    )
    # /pulls/{number}, /issues/{number}
    result = _re.sub(r"/(pulls|issues)/(\d+)", r"/\1/{number}", result)
    # /git/trees/{branch}
    result = _re.sub(r"/git/trees/[^/]+", "/git/trees/{ref}", result)
    # /contents/{path} (everything after /contents/)
    result = _re.sub(r"/contents/.+", "/contents/{path}", result)
    # /merge-upstream, /merge — leave as-is
    # /forks — leave as-is
    return result


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PullRequest:
    """Represents a GitHub pull request."""

    number: int
    html_url: str
    title: str = ""
    author: str = ""
    base_ref: str = ""
    head_ref: str = ""
    body: str = ""
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    node_id: str = ""


@dataclass
class RepoFile:
    """Represents a file in a GitHub repository."""

    path: str
    content: str = ""


@dataclass
class PRFile:
    """Represents a changed file in a pull request."""

    filename: str
    status: str = ""
    additions: int = 0
    deletions: int = 0


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GitHubAdapter:
    """Low-level GitHub REST API and local Git operations.

    Handles authentication, retries, and HTTP details.  Business logic
    belongs in higher-level modules (Planner, Implementer, Reviewer).
    """

    def __init__(
        self,
        token: str | None = None,
        api_url: str | None = None,
        git_user_name: str | None = None,
        git_user_email: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ):
        self.token = token or GITHUB_TOKEN
        self.api_url = (api_url or GITHUB_API_URL).rstrip("/")
        self.git_user_name = git_user_name or GIT_USER_NAME
        self.git_user_email = git_user_email or GIT_USER_EMAIL
        self.timeout = timeout
        self._session = self._create_session(max_retries, backoff_factor)

    def _create_session(
        self,
        max_retries: int,
        backoff_factor: float,
    ) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _headers(self, accept: str = "application/vnd.github.v3+json") -> dict[str, str]:
        """Build request headers with authentication."""
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN environment variable is not set")
        return {
            "Authorization": f"token {self.token}",
            "Accept": accept,
        }

    # ------------------------------------------------------------------
    # Core HTTP methods
    # ------------------------------------------------------------------

    @tracer.start_as_current_span("github.request")
    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json_body: dict | None = None,
        accept: str = "application/vnd.github.v3+json",
    ) -> requests.Response:
        """Execute an HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/repos/owner/repo/pulls")
            params: Query parameters
            json_body: JSON request body
            accept: Accept header value

        Returns:
            Response object

        Raises:
            requests.HTTPError: On non-2xx responses
        """
        span = trace.get_current_span()
        span.set_attribute("http.method", method)
        span.set_attribute("http.route", _normalise_github_route(endpoint))
        span.set_attribute("http.url", f"{self.api_url}{endpoint}")

        url = f"{self.api_url}{endpoint}"
        try:
            resp = self._session.request(
                method=method,
                url=url,
                headers=self._headers(accept),
                params=params,
                json=json_body,
                timeout=self.timeout,
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise

        span.set_attribute("http.status_code", resp.status_code)
        if resp.status_code >= 500:
            span.set_status(StatusCode.ERROR, f"HTTP {resp.status_code}")
        if not resp.ok:
            logger.warning(
                "Request failed: %s %s — HTTP %d: %.500s",
                method, endpoint, resp.status_code, resp.text,
            )
        resp.raise_for_status()
        return resp

    def _get(
        self,
        endpoint: str,
        params: dict | None = None,
        accept: str = "application/vnd.github.v3+json",
    ) -> requests.Response:
        """Execute a GET request."""
        return self._request("GET", endpoint, params=params, accept=accept)

    def _post(
        self,
        endpoint: str,
        json_body: dict | None = None,
    ) -> requests.Response:
        """Execute a POST request."""
        return self._request("POST", endpoint, json_body=json_body)

    def _put(
        self,
        endpoint: str,
        json_body: dict | None = None,
    ) -> requests.Response:
        """Execute a PUT request."""
        return self._request("PUT", endpoint, json_body=json_body)

    def _patch(
        self,
        endpoint: str,
        json_body: dict | None = None,
    ) -> requests.Response:
        """Execute a PATCH request."""
        return self._request("PATCH", endpoint, json_body=json_body)

    # ------------------------------------------------------------------
    # Pull request operations
    # ------------------------------------------------------------------

    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> PullRequest:
        """Fetch a pull request by number.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            PullRequest object
        """
        resp = self._get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
        data = resp.json()
        return PullRequest(
            number=data.get("number", pr_number),
            html_url=data.get("html_url", ""),
            title=data.get("title", ""),
            author=data.get("user", {}).get("login", ""),
            base_ref=data.get("base", {}).get("ref", ""),
            head_ref=data.get("head", {}).get("ref", ""),
            body=data.get("body", "") or "",
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
            changed_files=data.get("changed_files", 0),
            node_id=data.get("node_id", ""),
        )

    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Fetch the unified diff for a pull request.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            Unified diff as a string
        """
        resp = self._get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}",
            accept="application/vnd.github.v3.diff",
        )
        return resp.text

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[PRFile]:
        """Get the list of files changed in a pull request.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            List of PRFile objects
        """
        resp = self._get(f"/repos/{owner}/{repo}/pulls/{pr_number}/files")
        files = resp.json()
        return [
            PRFile(
                filename=f.get("filename", ""),
                status=f.get("status", ""),
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
            )
            for f in files
        ]

    def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: str = "",
        draft: bool = False,
    ) -> PullRequest:
        """Create a new pull request.

        Args:
            owner: Repository owner
            repo: Repository name
            title: PR title
            head: Head branch name
            base: Base branch name
            body: PR body (markdown)
            draft: If True, create the PR as a draft

        Returns:
            PullRequest object
        """
        resp = self._post(
            f"/repos/{owner}/{repo}/pulls",
            json_body={
                "title": title,
                "head": head,
                "base": base,
                "body": body,
                "draft": draft,
            },
        )
        data = resp.json()
        return PullRequest(
            number=data.get("number", 0),
            html_url=data.get("html_url", ""),
            title=data.get("title", title),
            base_ref=base,
            head_ref=head,
            body=body,
            node_id=data.get("node_id", ""),
        )

    def append_to_pr_body(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        text: str,
    ) -> None:
        """Append text to an existing PR's body."""
        pr = self.get_pull_request(owner, repo, pr_number)
        new_body = (pr.body or "") + text
        self._patch(
            f"/repos/{owner}/{repo}/pulls/{pr_number}",
            json_body={"body": new_body},
        )

    def mark_pr_ready_for_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> bool:
        """Convert a draft PR to ready for review via the GraphQL API.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            True if successfully marked as ready
        """
        pr = self.get_pull_request(owner, repo, pr_number)
        if not pr.node_id:
            logger.error("Cannot mark ready: no node_id for PR #%d", pr_number)
            return False

        query = """
            mutation($prId: ID!) {
                markPullRequestReadyForReview(input: {pullRequestId: $prId}) {
                    pullRequest { isDraft }
                }
            }
        """
        resp = self._session.post(
            f"{self.api_url}/graphql",
            headers=self._headers(),
            json={"query": query, "variables": {"prId": pr.node_id}},
            timeout=self.timeout,
        )
        if not resp.ok:
            logger.error(
                "GraphQL markPullRequestReadyForReview failed: HTTP %d: %.500s",
                resp.status_code, resp.text,
            )
            return False

        data = resp.json()
        if "errors" in data:
            logger.error("GraphQL errors: %s", data["errors"])
            return False

        logger.info("PR #%d marked as ready for review", pr_number)
        return True

    def merge_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        merge_method: str = "rebase",
    ) -> bool:
        """Merge a pull request.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            merge_method: Merge method (merge, squash, rebase)

        Returns:
            True if merged successfully
        """
        try:
            self._put(
                f"/repos/{owner}/{repo}/pulls/{pr_number}/merge",
                json_body={"merge_method": merge_method},
            )
            logger.info("PR #%d merged", pr_number)
            return True
        except requests.RequestException as exc:
            logger.error("Merge failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Issue / comment operations
    # ------------------------------------------------------------------

    def post_issue_comment(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        body: str,
    ) -> str | None:
        """Post a comment on an issue or pull request.

        Args:
            owner: Repository owner
            repo: Repository name
            issue_number: Issue or PR number
            body: Comment body (markdown)

        Returns:
            The comment's html_url, or None on failure
        """
        try:
            resp = self._post(
                f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
                json_body={"body": body},
            )
            comment_url = resp.json().get("html_url")
            logger.info("Comment posted: %s", comment_url)
            return comment_url
        except requests.RequestException as exc:
            logger.error("Failed to post comment: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Repository operations
    # ------------------------------------------------------------------

    def can_push(self, owner: str, repo: str) -> bool:
        """Check whether the authenticated user can push to a repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            True if the token grants push access
        """
        resp = self._get(f"/repos/{owner}/{repo}")
        return resp.json().get("permissions", {}).get("push", False)

    def get_authenticated_user(self) -> str:
        """Return the login of the authenticated user.

        Returns:
            GitHub username string
        """
        resp = self._get("/user")
        return resp.json()["login"]

    def fork_repo(self, owner: str, repo: str) -> str:
        """Fork a repository into the authenticated user's account.

        If the fork already exists GitHub returns it immediately.

        Args:
            owner: Upstream repository owner
            repo: Upstream repository name

        Returns:
            Clone URL of the fork (HTTPS)
        """
        resp = self._post(f"/repos/{owner}/{repo}/forks")
        data = resp.json()
        fork_url = data.get("clone_url", "")
        fork_owner = data.get("owner", {}).get("login", "")
        logger.info("Fork ready: %s/%s -> %s", owner, repo, fork_owner)

        # GitHub may return 202 while the fork is still being created.
        # Poll until the fork is accessible (up to ~30 s).
        for _ in range(6):
            try:
                self._get(f"/repos/{fork_owner}/{repo}")
                break
            except requests.HTTPError:
                time.sleep(5)

        return fork_url

    def sync_fork(self, owner: str, repo: str, branch: str) -> bool:
        """Sync a fork's branch with its upstream parent.

        Uses the GitHub "merge upstream" API so the fork's default
        branch stays up-to-date before we branch off it.

        Args:
            owner: Fork owner (typically the bot user)
            repo: Repository name
            branch: Branch to sync (usually the default branch)

        Returns:
            True if the sync succeeded or was already up-to-date
        """
        try:
            self._post(
                f"/repos/{owner}/{repo}/merge-upstream",
                json_body={"branch": branch},
            )
            logger.info("Fork %s/%s synced on branch %s", owner, repo, branch)
            return True
        except requests.HTTPError as exc:
            # 409 means "already up to date" — that's fine.
            if exc.response is not None and exc.response.status_code == 409:
                logger.info("Fork %s/%s already up-to-date", owner, repo)
                return True
            logger.error("Fork sync failed: %s", exc)
            return False

    def get_repo_default_branch(self, owner: str, repo: str) -> str:
        """Get the default branch of a repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Default branch name (e.g., "main")
        """
        resp = self._get(f"/repos/{owner}/{repo}")
        return resp.json().get("default_branch", "main")

    def get_repo_tree(self, owner: str, repo: str, branch: str) -> list[str]:
        """Get all file paths in a repository tree.

        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name

        Returns:
            List of file paths
        """
        resp = self._get(
            f"/repos/{owner}/{repo}/git/trees/{branch}",
            params={"recursive": "1"},
        )
        data = resp.json()
        return [
            item["path"]
            for item in data.get("tree", [])
            if item.get("type") == "blob"
        ]

    def get_file_contents(self, owner: str, repo: str, path: str) -> str:
        """Read a file from the repository via the Contents API.

        Args:
            owner: Repository owner
            repo: Repository name
            path: File path within the repository

        Returns:
            File contents as a string
        """
        resp = self._get(f"/repos/{owner}/{repo}/contents/{path}")
        data = resp.json()
        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        dl = data.get("download_url")
        if dl:
            return requests.get(dl, timeout=self.timeout).text
        return ""

    # ------------------------------------------------------------------
    # Local git operations
    # ------------------------------------------------------------------

    def _run_git(self, cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
        """Run a git command, redacting any token from errors.

        ``subprocess.CalledProcessError`` embeds the full command in its
        string representation.  If the command contains an auth URL the
        token would leak into logs and stack traces.  This wrapper
        catches the exception and re-raises with a sanitised command.
        """
        try:
            return subprocess.run(
                cmd, cwd=cwd, check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise subprocess.CalledProcessError(
                exc.returncode,
                [self._redact_token(a) for a in exc.cmd],
                output=self._redact_token(exc.output or ""),
                stderr=self._redact_token(exc.stderr or ""),
            ) from None

    def _redact_token(self, text: str) -> str:
        """Replace the current token with ``***`` in *text*."""
        if self.token and self.token in text:
            return text.replace(self.token, "***")
        return text

    @tracer.start_as_current_span("github.clone_repo")
    def clone_repo(self, repo_url: str, dest: Path) -> str:
        """Clone a repository to a local directory.

        Removes any existing clone at *dest* first.  Injects token auth
        for github.com URLs.

        Args:
            repo_url: Repository URL (HTTPS)
            dest: Destination directory

        Returns:
            Absolute path to the cloned repository
        """
        span = trace.get_current_span()
        span.set_attribute("repo.url", repo_url)
        span.set_attribute("repo.dest", str(dest))
        if dest.exists():
            logger.info("Removing existing clone at %s", dest)
            shutil.rmtree(dest)

        dest.parent.mkdir(parents=True, exist_ok=True)
        auth_url = self._build_auth_url(repo_url)

        self._run_git(["git", "clone", auth_url, str(dest)])
        return str(dest)

    @tracer.start_as_current_span("github.create_branch")
    def create_branch(self, repo_path: str, branch_name: str) -> str:
        """Create and checkout a new branch in a local repository.

        Args:
            repo_path: Path to the local repository
            branch_name: Name of the branch to create

        Returns:
            The branch name
        """
        span = trace.get_current_span()
        span.set_attribute("branch.name", branch_name)
        self._run_git(["git", "checkout", "-b", branch_name], cwd=repo_path)
        return branch_name

    @tracer.start_as_current_span("github.commit_and_push")
    def commit_and_push(
        self,
        repo_path: str,
        repo_url: str,
        branch_name: str,
        commit_msg: str,
    ) -> str:
        """Stage all changes, commit, and push to remote.

        Configures git user.name/email, runs ``git add -A``, commits,
        and pushes with token auth.

        Args:
            repo_path: Path to the local repository
            repo_url: Remote repository URL
            branch_name: Branch to push
            commit_msg: Commit message

        Returns:
            URL to the pushed branch

        Raises:
            RuntimeError: If there are no staged changes
        """
        span = trace.get_current_span()
        span.set_attribute("repo.url", repo_url)
        span.set_attribute("branch.name", branch_name)
        run = lambda cmd: self._run_git(cmd, cwd=repo_path)

        run(["git", "config", "user.name", self.git_user_name])
        run(["git", "config", "user.email", self.git_user_email])
        run(["git", "add", "-A"])

        # Check if there are staged changes before committing
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_path, capture_output=True,
        )
        if status.returncode == 0:
            raise RuntimeError(
                "No files were changed by the implementation steps. "
                "Nothing to commit."
            )

        run(["git", "commit", "-m", commit_msg])

        if self.token and "github.com" in repo_url:
            auth_url = self._build_auth_url(repo_url)
            run(["git", "push", auth_url, branch_name])
        else:
            run(["git", "push", "origin", branch_name])

        base = repo_url.rstrip("/")
        if base.endswith(".git"):
            base = base[:-4]
        return f"{base}/tree/{branch_name}"

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def _build_auth_url(self, repo_url: str) -> str:
        """Inject token into a github.com HTTPS URL for authentication."""
        if self.token and "github.com" in repo_url:
            return repo_url.replace(
                "https://github.com",
                f"https://{self.token}@github.com",
            )
        return repo_url

    @staticmethod
    def parse_pr_url(pr_url: str) -> tuple[str, str, int]:
        """Parse a GitHub PR URL into (owner, repo, pr_number).

        Args:
            pr_url: e.g., "https://github.com/owner/repo/pull/123"

        Returns:
            Tuple of (owner, repo, pr_number)

        Raises:
            ValueError: If the URL does not match the expected pattern
        """
        m = _PR_URL_RE.search(pr_url)
        if not m:
            raise ValueError(
                f"Invalid GitHub PR URL: {pr_url!r}. "
                "Expected: https://github.com/OWNER/REPO/pull/NUMBER"
            )
        return m.group("owner"), m.group("repo"), int(m.group("number"))

    @staticmethod
    def parse_repo_url(repo_url: str) -> tuple[str, str]:
        """Parse a GitHub repo URL into (owner, repo).

        Args:
            repo_url: e.g., "https://github.com/owner/repo" or
                      "https://github.com/owner/repo.git"

        Returns:
            Tuple of (owner, repo)

        Raises:
            ValueError: If the URL cannot be parsed
        """
        url_clean = repo_url.rstrip("/")
        if url_clean.endswith(".git"):
            url_clean = url_clean[:-4]
        parsed = urlparse(url_clean)
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Cannot extract owner/repo from: {repo_url}")
        return parts[0], parts[1]
