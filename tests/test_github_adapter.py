"""Tests for stary.github_adapter – GitHub REST API and local Git operations."""

import base64
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from stary.github_adapter import (
    GitHubAdapter,
    PRFile,
    PullRequest,
    RepoFile,
)
from stary.config import RepoAllowlist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(**kwargs) -> GitHubAdapter:
    """Create an adapter with a fake token (no real API calls)."""
    defaults = {"token": "ghp_test_token_123"}
    defaults.update(kwargs)
    return GitHubAdapter(**defaults)


def _mock_response(json_data=None, text="", status_code=200, ok=True):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.ok = ok
    resp.text = text
    resp.json.return_value = json_data or {}
    resp.raise_for_status.return_value = None
    return resp


def _successful_run(cmd, **kwargs):
    """Default side-effect for subprocess.run that always succeeds."""
    result = MagicMock()
    result.returncode = 0
    return result


def _run_with_changes(cmd, **kwargs):
    """Side-effect where git diff --cached reports changes exist."""
    result = MagicMock()
    if cmd == ["git", "diff", "--cached", "--quiet"]:
        result.returncode = 1
    else:
        result.returncode = 0
    return result


def _run_no_changes(cmd, **kwargs):
    """Side-effect where git diff --cached reports no changes."""
    result = MagicMock()
    result.returncode = 0
    return result


# ---------------------------------------------------------------------------
# Constructor / configuration
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_defaults(self):
        adapter = _make_adapter()
        assert adapter.token == "ghp_test_token_123"
        assert adapter.api_url == "https://api.github.com"

    def test_custom_api_url_strips_trailing_slash(self):
        adapter = _make_adapter(api_url="https://gh.corp.com/api/v3/")
        assert adapter.api_url == "https://gh.corp.com/api/v3"

    def test_custom_git_identity(self):
        adapter = _make_adapter(
            git_user_name="botuser",
            git_user_email="bot@example.com",
        )
        assert adapter.git_user_name == "botuser"
        assert adapter.git_user_email == "bot@example.com"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestHeaders:
    def test_headers_include_token(self):
        adapter = _make_adapter()
        headers = adapter._headers()
        assert headers["Authorization"] == "token ghp_test_token_123"
        assert "Accept" in headers

    def test_headers_custom_accept(self):
        adapter = _make_adapter()
        headers = adapter._headers(accept="application/vnd.github.v3.diff")
        assert headers["Accept"] == "application/vnd.github.v3.diff"

    def test_headers_raises_without_token(self):
        adapter = _make_adapter(token="")
        adapter.token = ""
        with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
            adapter._headers()


# ---------------------------------------------------------------------------
# Pull request operations
# ---------------------------------------------------------------------------


class TestGetPullRequest:
    def test_parses_response(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={
            "number": 42,
            "html_url": "https://github.com/owner/repo/pull/42",
            "title": "Fix bug",
            "user": {"login": "alice"},
            "base": {"ref": "main"},
            "head": {"ref": "fix/bug"},
            "body": "Fixes #10",
            "additions": 5,
            "deletions": 2,
            "changed_files": 1,
        })

        pr = adapter.get_pull_request("owner", "repo", 42)

        assert isinstance(pr, PullRequest)
        assert pr.number == 42
        assert pr.title == "Fix bug"
        assert pr.author == "alice"
        assert pr.base_ref == "main"
        assert pr.head_ref == "fix/bug"
        assert pr.additions == 5
        assert pr.deletions == 2
        assert pr.changed_files == 1

    def test_handles_null_body(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={
            "number": 1,
            "html_url": "",
            "body": None,
        })

        pr = adapter.get_pull_request("o", "r", 1)
        assert pr.body == ""


class TestGetPrDiff:
    def test_returns_diff_text(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        diff_text = "diff --git a/file.py b/file.py\n+new line"
        adapter._session.request.return_value = _mock_response(text=diff_text)

        result = adapter.get_pr_diff("owner", "repo", 1)

        assert result == diff_text


class TestGetPrFiles:
    def test_parses_file_list(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data=[
            {"filename": "src/main.py", "status": "modified", "additions": 3, "deletions": 1},
            {"filename": "tests/test.py", "status": "added", "additions": 10, "deletions": 0},
        ])

        files = adapter.get_pr_files("owner", "repo", 1)

        assert len(files) == 2
        assert isinstance(files[0], PRFile)
        assert files[0].filename == "src/main.py"
        assert files[0].status == "modified"
        assert files[1].additions == 10

    def test_empty_pr(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data=[])

        files = adapter.get_pr_files("owner", "repo", 1)
        assert files == []


class TestCreatePullRequest:
    def test_creates_and_returns_pr(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={
            "number": 99,
            "html_url": "https://github.com/owner/repo/pull/99",
            "title": "PROJ-1 fix stuff",
        })

        pr = adapter.create_pull_request(
            owner="owner",
            repo="repo",
            title="PROJ-1 fix stuff",
            head="dev/bot/PROJ-1",
            base="master",
            body="Automated PR",
        )

        assert pr.number == 99
        assert pr.html_url == "https://github.com/owner/repo/pull/99"
        # Verify the POST payload
        call_args = adapter._session.request.call_args
        assert call_args.kwargs["method"] == "POST" or call_args[1].get("method") == "POST"


class TestMergePullRequest:
    def test_merge_success(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={
            "merged": True,
        })

        result = adapter.merge_pull_request("owner", "repo", 42)
        assert result is True

    def test_merge_failure_returns_false(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        resp = _mock_response(status_code=405, ok=False)
        resp.raise_for_status.side_effect = requests.HTTPError("Not allowed")
        adapter._session.request.return_value = resp

        result = adapter.merge_pull_request("owner", "repo", 42)
        assert result is False


# ---------------------------------------------------------------------------
# Issue / comment operations
# ---------------------------------------------------------------------------


class TestPostIssueComment:
    def test_returns_comment_url(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={
            "html_url": "https://github.com/owner/repo/issues/1#comment-123",
        })

        url = adapter.post_issue_comment("owner", "repo", 1, "Nice work!")
        assert url == "https://github.com/owner/repo/issues/1#comment-123"

    def test_returns_none_on_failure(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        resp = _mock_response(status_code=403, ok=False)
        resp.raise_for_status.side_effect = requests.HTTPError("Forbidden")
        adapter._session.request.return_value = resp

        url = adapter.post_issue_comment("owner", "repo", 1, "body")
        assert url is None


# ---------------------------------------------------------------------------
# Repository operations
# ---------------------------------------------------------------------------


class TestGetRepoDefaultBranch:
    def test_returns_default_branch(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={
            "default_branch": "develop",
        })

        branch = adapter.get_repo_default_branch("owner", "repo")
        assert branch == "develop"

    def test_falls_back_to_main(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={})

        branch = adapter.get_repo_default_branch("owner", "repo")
        assert branch == "main"


class TestCanPush:
    def test_returns_true_when_push_allowed(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={
            "permissions": {"admin": False, "push": True, "pull": True},
        })

        assert adapter.can_push("owner", "repo") is True

    def test_returns_false_when_push_denied(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={
            "permissions": {"admin": False, "push": False, "pull": True},
        })

        assert adapter.can_push("owner", "repo") is False

    def test_returns_false_when_permissions_missing(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={})

        assert adapter.can_push("owner", "repo") is False


class TestGetAuthenticatedUser:
    def test_returns_login(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={
            "login": "botuser",
        })

        assert adapter.get_authenticated_user() == "botuser"


class TestForkRepo:
    def test_creates_fork_and_returns_clone_url(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()

        # First call: POST /repos/owner/repo/forks
        fork_resp = _mock_response(json_data={
            "clone_url": "https://github.com/botuser/repo.git",
            "owner": {"login": "botuser"},
        })
        # Second call: GET /repos/botuser/repo (fork ready check)
        ready_resp = _mock_response(json_data={"full_name": "botuser/repo"})
        adapter._session.request.side_effect = [fork_resp, ready_resp]

        url = adapter.fork_repo("owner", "repo")

        assert url == "https://github.com/botuser/repo.git"
        assert adapter._session.request.call_count == 2


class TestSyncFork:
    def test_sync_success(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={
            "merge_type": "fast-forward",
        })

        assert adapter.sync_fork("botuser", "repo", "main") is True

    def test_sync_already_up_to_date(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        resp_409 = _mock_response(status_code=409, ok=False)
        http_err = requests.HTTPError("Conflict")
        http_err.response = resp_409
        resp_409.raise_for_status.side_effect = http_err
        adapter._session.request.return_value = resp_409

        assert adapter.sync_fork("botuser", "repo", "main") is True

    def test_sync_failure_returns_false(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        resp_500 = _mock_response(status_code=500, ok=False)
        http_err = requests.HTTPError("Server Error")
        http_err.response = resp_500
        resp_500.raise_for_status.side_effect = http_err
        adapter._session.request.return_value = resp_500

        assert adapter.sync_fork("botuser", "repo", "main") is False


class TestGetRepoTree:
    def test_returns_blob_paths_only(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={
            "tree": [
                {"path": "src/main.py", "type": "blob"},
                {"path": "src", "type": "tree"},
                {"path": "README.md", "type": "blob"},
            ],
        })

        paths = adapter.get_repo_tree("owner", "repo", "main")
        assert paths == ["src/main.py", "README.md"]


class TestGetFileContents:
    def test_decodes_base64(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        content = "print('hello')\n"
        encoded = base64.b64encode(content.encode()).decode()
        adapter._session.request.return_value = _mock_response(json_data={
            "encoding": "base64",
            "content": encoded,
        })

        result = adapter.get_file_contents("owner", "repo", "main.py")
        assert result == content

    def test_falls_back_to_download_url(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={
            "download_url": "https://raw.githubusercontent.com/owner/repo/main/file.txt",
        })

        with patch("stary.github_adapter.requests.get") as mock_get:
            mock_get.return_value = _mock_response(text="file content")
            result = adapter.get_file_contents("owner", "repo", "file.txt")

        assert result == "file content"

    def test_returns_empty_string_when_no_content(self):
        adapter = _make_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(json_data={})

        result = adapter.get_file_contents("owner", "repo", "empty.txt")
        assert result == ""


# ---------------------------------------------------------------------------
# Local git operations
# ---------------------------------------------------------------------------


class TestCloneRepo:
    @patch("stary.github_adapter.subprocess.run")
    @patch("stary.github_adapter.shutil.rmtree")
    def test_clones_with_token_auth(self, mock_rmtree, mock_run):
        adapter = _make_adapter()
        dest = Path("/tmp/test_clone/myrepo")

        with patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "mkdir"):
            result = adapter.clone_repo("https://github.com/owner/repo.git", dest)

        assert result == str(dest)
        clone_cmd = mock_run.call_args[0][0]
        assert clone_cmd[0] == "git"
        assert clone_cmd[1] == "clone"
        assert "ghp_test_token_123@github.com" in clone_cmd[2]

    @patch("stary.github_adapter.subprocess.run")
    @patch("stary.github_adapter.shutil.rmtree")
    def test_removes_existing_clone(self, mock_rmtree, mock_run):
        adapter = _make_adapter()
        dest = Path("/tmp/test_clone/myrepo")

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "mkdir"):
            adapter.clone_repo("https://github.com/owner/repo.git", dest)

        mock_rmtree.assert_called_once_with(dest)

    @patch("stary.github_adapter.subprocess.run")
    def test_non_github_url_no_token_injection(self, mock_run):
        adapter = _make_adapter()
        dest = Path("/tmp/test_clone/myrepo")

        with patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "mkdir"):
            adapter.clone_repo("https://gitlab.com/owner/repo.git", dest)

        clone_cmd = mock_run.call_args[0][0]
        assert "ghp_test_token_123" not in clone_cmd[2]

    @patch("stary.github_adapter.subprocess.run")
    def test_clone_failure_redacts_token(self, mock_run):
        adapter = _make_adapter()
        dest = Path("/tmp/test_clone/myrepo")

        mock_run.side_effect = subprocess.CalledProcessError(
            128,
            ["git", "clone", "https://ghp_test_token_123@github.com/owner/repo.git", "/tmp/x"],
            output="",
            stderr="fatal: repo not found https://ghp_test_token_123@github.com/owner/repo.git",
        )

        with patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "mkdir"):
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                adapter.clone_repo("https://github.com/owner/repo.git", dest)

        exc = exc_info.value
        # Token MUST NOT appear anywhere in the re-raised exception
        assert "ghp_test_token_123" not in str(exc)
        assert "ghp_test_token_123" not in str(exc.cmd)
        assert "ghp_test_token_123" not in (exc.stderr or "")
        assert "ghp_test_token_123" not in (exc.stdout or "")
        # But the redacted placeholder should be there
        assert "***" in str(exc.cmd)


class TestCreateBranch:
    @patch("stary.github_adapter.subprocess.run")
    def test_creates_branch(self, mock_run):
        adapter = _make_adapter()

        result = adapter.create_branch("/tmp/repo", "dev/bot/PROJ-1")

        assert result == "dev/bot/PROJ-1"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["git", "checkout", "-b", "dev/bot/PROJ-1"]


class TestCommitAndPush:
    @patch("stary.github_adapter.subprocess.run")
    def test_full_commit_push_flow(self, mock_run):
        adapter = _make_adapter()
        mock_run.side_effect = _run_with_changes

        url = adapter.commit_and_push(
            repo_path="/tmp/repo",
            repo_url="https://github.com/owner/repo.git",
            branch_name="dev/bot/PROJ-1",
            commit_msg="PROJ-1 fix stuff",
        )

        assert url == "https://github.com/owner/repo/tree/dev/bot/PROJ-1"

        # Verify git config, add, commit, push were called
        calls = [c[0][0] for c in mock_run.call_args_list]
        assert ["git", "config", "user.name", "qaplatformbot"] in calls
        assert ["git", "config", "user.email", "sys_qaplatformbot@intel.com"] in calls
        assert ["git", "add", "-A"] in calls
        assert ["git", "commit", "-m", "PROJ-1 fix stuff"] in calls

    @patch("stary.github_adapter.subprocess.run")
    def test_raises_when_nothing_to_commit(self, mock_run):
        adapter = _make_adapter()
        mock_run.side_effect = _run_no_changes

        with pytest.raises(RuntimeError, match="Nothing to commit"):
            adapter.commit_and_push(
                repo_path="/tmp/repo",
                repo_url="https://github.com/owner/repo",
                branch_name="dev/bot/X",
                commit_msg="msg",
            )

    @patch("stary.github_adapter.subprocess.run")
    def test_push_uses_origin_for_non_github(self, mock_run):
        adapter = _make_adapter()
        mock_run.side_effect = _run_with_changes

        adapter.commit_and_push(
            repo_path="/tmp/repo",
            repo_url="https://gitlab.com/owner/repo",
            branch_name="feature",
            commit_msg="msg",
        )

        push_call = [c[0][0] for c in mock_run.call_args_list
                     if c[0][0][0:2] == ["git", "push"]]
        assert len(push_call) == 1
        assert push_call[0] == ["git", "push", "origin", "feature"]

    @patch("stary.github_adapter.subprocess.run")
    def test_push_failure_redacts_token(self, mock_run):
        adapter = _make_adapter()
        token = "ghp_test_token_123"
        auth_url = f"https://{token}@github.com/owner/repo.git"

        call_count = 0

        def side_effect(cmd, **kwargs):
            nonlocal call_count
            if cmd == ["git", "diff", "--cached", "--quiet"]:
                r = MagicMock()
                r.returncode = 1
                return r
            # Fail on the push command
            if len(cmd) >= 2 and cmd[0] == "git" and cmd[1] == "push":
                raise subprocess.CalledProcessError(
                    1, cmd,
                    output="",
                    stderr=f"fatal: auth failed for {auth_url}",
                )
            r = MagicMock()
            r.returncode = 0
            return r

        mock_run.side_effect = side_effect

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            adapter.commit_and_push(
                repo_path="/tmp/repo",
                repo_url="https://github.com/owner/repo.git",
                branch_name="feature",
                commit_msg="msg",
            )

        exc = exc_info.value
        assert token not in str(exc)
        assert token not in str(exc.cmd)
        assert token not in (exc.stderr or "")
        assert "***" in str(exc.cmd)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


class TestParsePrUrl:
    def test_valid_url(self):
        owner, repo, number = GitHubAdapter.parse_pr_url(
            "https://github.com/myorg/myrepo/pull/123"
        )
        assert owner == "myorg"
        assert repo == "myrepo"
        assert number == 123

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Invalid GitHub PR URL"):
            GitHubAdapter.parse_pr_url("https://gitlab.com/org/repo/merge/1")

    def test_url_embedded_in_text(self):
        owner, repo, number = GitHubAdapter.parse_pr_url(
            "See https://github.com/org/repo/pull/7 for details"
        )
        assert owner == "org"
        assert number == 7


class TestParseRepoUrl:
    def test_https_url(self):
        owner, repo = GitHubAdapter.parse_repo_url("https://github.com/owner/repo")
        assert owner == "owner"
        assert repo == "repo"

    def test_git_suffix(self):
        owner, repo = GitHubAdapter.parse_repo_url("https://github.com/owner/repo.git")
        assert owner == "owner"
        assert repo == "repo"

    def test_trailing_slash(self):
        owner, repo = GitHubAdapter.parse_repo_url("https://github.com/owner/repo/")
        assert owner == "owner"
        assert repo == "repo"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Cannot extract owner/repo"):
            GitHubAdapter.parse_repo_url("https://github.com/only-owner")


class TestBuildAuthUrl:
    def test_injects_token_for_github(self):
        adapter = _make_adapter()
        url = adapter._build_auth_url("https://github.com/owner/repo.git")
        assert url == "https://ghp_test_token_123@github.com/owner/repo.git"

    def test_no_injection_for_non_github(self):
        adapter = _make_adapter()
        url = adapter._build_auth_url("https://gitlab.com/owner/repo.git")
        assert url == "https://gitlab.com/owner/repo.git"

    def test_no_injection_without_token(self):
        adapter = _make_adapter(token="")
        adapter.token = ""
        url = adapter._build_auth_url("https://github.com/owner/repo.git")
        assert url == "https://github.com/owner/repo.git"


# ---------------------------------------------------------------------------
# Token redaction
# ---------------------------------------------------------------------------


class TestRedactToken:
    def test_replaces_token_with_stars(self):
        adapter = _make_adapter()
        result = adapter._redact_token(
            "https://ghp_test_token_123@github.com/owner/repo.git"
        )
        assert "ghp_test_token_123" not in result
        assert "***@github.com" in result

    def test_no_op_when_token_absent(self):
        adapter = _make_adapter()
        text = "nothing secret here"
        assert adapter._redact_token(text) == text

    def test_no_op_when_token_is_empty(self):
        adapter = _make_adapter(token="")
        adapter.token = ""
        assert adapter._redact_token("any text ghp_xxx") == "any text ghp_xxx"

    def test_redacts_multiple_occurrences(self):
        adapter = _make_adapter()
        text = "ghp_test_token_123 and ghp_test_token_123 again"
        result = adapter._redact_token(text)
        assert "ghp_test_token_123" not in result
        assert result.count("***") == 2


class TestRunGit:
    @patch("stary.github_adapter.subprocess.run")
    def test_success_passes_through(self, mock_run):
        adapter = _make_adapter()
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        result = adapter._run_git(["git", "status"], cwd="/tmp")
        assert result.returncode == 0

    @patch("stary.github_adapter.subprocess.run")
    def test_failure_redacts_cmd_and_stderr(self, mock_run):
        adapter = _make_adapter()
        token = "ghp_test_token_123"
        auth_url = f"https://{token}@github.com/owner/repo.git"
        cmd = ["git", "clone", auth_url, "/tmp/dest"]

        mock_run.side_effect = subprocess.CalledProcessError(
            128, cmd, output="", stderr=f"fatal: could not read from {auth_url}",
        )

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            adapter._run_git(cmd)

        exc = exc_info.value
        # Token must be completely absent from the raised exception
        assert token not in str(exc.cmd)
        assert token not in (exc.stdout or "")
        assert token not in (exc.stderr or "")
        assert token not in str(exc)
        # Redacted command still has the structure
        assert "***@github.com" in str(exc.cmd)

    @patch("stary.github_adapter.subprocess.run")
    def test_failure_without_token_in_cmd_unchanged(self, mock_run):
        adapter = _make_adapter()
        cmd = ["git", "checkout", "-b", "feature"]

        mock_run.side_effect = subprocess.CalledProcessError(
            1, cmd, output="", stderr="error: branch already exists",
        )

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            adapter._run_git(cmd)

        exc = exc_info.value
        assert exc.cmd == cmd
        assert exc.stderr == "error: branch already exists"


# ---------------------------------------------------------------------------
# Repo-allowlist guard on write methods
# ---------------------------------------------------------------------------

class TestRepoAllowlistGuard:
    """Verify that write operations raise when the repo is not allowed."""

    def _blocked_adapter(self) -> GitHubAdapter:
        return _make_adapter(repo_allowlist=RepoAllowlist(["allowed-org/*"]))

    def _open_adapter(self) -> GitHubAdapter:
        """Adapter with no allowlist — everything goes through."""
        return _make_adapter()

    # -- create_pull_request -------------------------------------------------

    @patch("stary.github_adapter.requests.Session.post")
    def test_create_pr_blocked(self, _mock_post):
        adapter = self._blocked_adapter()
        with pytest.raises(ValueError, match="not in ALLOWED_REPOS"):
            adapter.create_pull_request("evil-org", "repo", "t", "h", "b")

    def test_create_pr_allowed(self):
        adapter = self._blocked_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(
            json_data={"number": 1, "html_url": "https://github.com/allowed-org/repo/pull/1", "node_id": "x"},
        )
        pr = adapter.create_pull_request("allowed-org", "repo", "title", "head", "base")
        assert pr.number == 1

    # -- fork_repo -----------------------------------------------------------

    @patch("stary.github_adapter.requests.Session.post")
    def test_fork_blocked(self, _mock_post):
        adapter = self._blocked_adapter()
        with pytest.raises(ValueError, match="not in ALLOWED_REPOS"):
            adapter.fork_repo("evil-org", "repo")

    # -- merge_pull_request --------------------------------------------------

    @patch("stary.github_adapter.requests.Session.put")
    def test_merge_blocked(self, _mock_put):
        adapter = self._blocked_adapter()
        with pytest.raises(ValueError, match="not in ALLOWED_REPOS"):
            adapter.merge_pull_request("evil-org", "repo", 1)

    # -- mark_pr_ready_for_review -------------------------------------------

    @patch("stary.github_adapter.requests.Session.get")
    def test_mark_ready_blocked(self, _mock_get):
        adapter = self._blocked_adapter()
        with pytest.raises(ValueError, match="not in ALLOWED_REPOS"):
            adapter.mark_pr_ready_for_review("evil-org", "repo", 1)

    # -- clone_repo ----------------------------------------------------------

    @patch("stary.github_adapter.subprocess.run")
    @patch("stary.github_adapter.shutil.rmtree")
    def test_clone_blocked(self, _mock_rm, _mock_run):
        adapter = self._blocked_adapter()
        from pathlib import Path
        with pytest.raises(ValueError, match="not in ALLOWED_REPOS"):
            adapter.clone_repo("https://github.com/evil-org/repo", Path("/tmp/dest"))

    # -- commit_and_push -----------------------------------------------------

    @patch("stary.github_adapter.subprocess.run")
    def test_push_blocked(self, _mock_run):
        adapter = self._blocked_adapter()
        with pytest.raises(ValueError, match="not in ALLOWED_REPOS"):
            adapter.commit_and_push("/tmp/repo", "https://github.com/evil-org/repo", "branch", "msg")

    # -- no allowlist (None) — everything allowed ----------------------------

    def test_no_allowlist_allows_all(self):
        adapter = self._open_adapter()
        adapter._session = MagicMock()
        adapter._session.request.return_value = _mock_response(
            json_data={"number": 1, "html_url": "https://github.com/any/repo/pull/1", "node_id": "x"},
        )
        pr = adapter.create_pull_request("any", "repo", "title", "head", "base")
        assert pr.number == 1
