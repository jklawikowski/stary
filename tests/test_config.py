"""Tests for stary.config – URL construction and validation helpers."""

import os
from unittest import mock

import pytest

from stary.config import build_dagster_run_url, get_dagster_base_url, validate_and_normalize_url
from stary.config import RepoAllowlist, get_repo_allowlist


# ---------------------------------------------------------------------------
# validate_and_normalize_url
# ---------------------------------------------------------------------------

class TestValidateAndNormalizeUrl:
    def test_strips_trailing_slash(self):
        assert validate_and_normalize_url("https://dagster.example.com/") == "https://dagster.example.com"

    def test_strips_multiple_trailing_slashes(self):
        assert validate_and_normalize_url("https://dagster.example.com///") == "https://dagster.example.com"

    def test_strips_whitespace(self):
        assert validate_and_normalize_url("  https://dagster.example.com  ") == "https://dagster.example.com"

    def test_valid_http(self):
        assert validate_and_normalize_url("http://localhost:3000") == "http://localhost:3000"

    def test_valid_https_with_path(self):
        assert validate_and_normalize_url("https://dagster.corp.com/prefix") == "https://dagster.corp.com/prefix"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            validate_and_normalize_url("")

    def test_rejects_non_http(self):
        with pytest.raises(ValueError):
            validate_and_normalize_url("ftp://dagster.example.com")

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            validate_and_normalize_url("not-a-url")


# ---------------------------------------------------------------------------
# build_dagster_run_url
# ---------------------------------------------------------------------------

class TestBuildDagsterRunUrl:
    def test_basic(self):
        url = build_dagster_run_url("https://dagster.example.com", "abc123")
        assert url == "https://dagster.example.com/runs/abc123"

    def test_trailing_slash_in_base(self):
        url = build_dagster_run_url("https://dagster.example.com/", "abc123")
        assert url == "https://dagster.example.com/runs/abc123"

    def test_none_base_url(self):
        assert build_dagster_run_url(None, "abc123") is None

    def test_empty_base_url(self):
        assert build_dagster_run_url("", "abc123") is None

    def test_none_run_id(self):
        assert build_dagster_run_url("https://dagster.example.com", None) is None

    def test_empty_run_id(self):
        assert build_dagster_run_url("https://dagster.example.com", "") is None

    def test_both_none(self):
        assert build_dagster_run_url(None, None) is None


# ---------------------------------------------------------------------------
# get_dagster_base_url
# ---------------------------------------------------------------------------

class TestGetDagsterBaseUrl:
    def test_returns_none_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert get_dagster_base_url() is None

    def test_returns_none_for_empty_string(self):
        with mock.patch.dict(os.environ, {"DAGSTER_BASE_URL": ""}):
            assert get_dagster_base_url() is None

    def test_returns_normalized_url(self):
        with mock.patch.dict(os.environ, {"DAGSTER_BASE_URL": "https://dagster.example.com/"}):
            assert get_dagster_base_url() == "https://dagster.example.com"

    def test_raises_on_invalid_url(self):
        with mock.patch.dict(os.environ, {"DAGSTER_BASE_URL": "not-a-url"}):
            with pytest.raises(ValueError):
                get_dagster_base_url()


# ---------------------------------------------------------------------------
# RepoAllowlist
# ---------------------------------------------------------------------------

class TestRepoAllowlist:
    def test_exact_match(self):
        al = RepoAllowlist(["myorg/myrepo"])
        assert al.is_allowed("myorg", "myrepo") is True

    def test_exact_no_match(self):
        al = RepoAllowlist(["myorg/myrepo"])
        assert al.is_allowed("myorg", "other") is False

    def test_org_wildcard(self):
        al = RepoAllowlist(["myorg/*"])
        assert al.is_allowed("myorg", "any-repo") is True
        assert al.is_allowed("otherorg", "any-repo") is False

    def test_case_insensitive(self):
        al = RepoAllowlist(["MyOrg/MyRepo"])
        assert al.is_allowed("myorg", "myrepo") is True
        assert al.is_allowed("MYORG", "MYREPO") is True

    def test_multiple_patterns(self):
        al = RepoAllowlist(["orgA/*", "orgB/specific-repo"])
        assert al.is_allowed("orgA", "anything") is True
        assert al.is_allowed("orgB", "specific-repo") is True
        assert al.is_allowed("orgB", "other") is False
        assert al.is_allowed("orgC", "whatever") is False

    def test_empty_denies_all(self):
        al = RepoAllowlist([])
        assert al.is_allowed("any", "repo") is False

    def test_whitespace_stripped(self):
        al = RepoAllowlist(["  myorg/myrepo  ", " orgB/* "])
        assert al.is_allowed("myorg", "myrepo") is True
        assert al.is_allowed("orgB", "x") is True

    def test_assert_allowed_passes(self):
        al = RepoAllowlist(["myorg/*"])
        al.assert_allowed("myorg", "repo")  # should not raise

    def test_assert_allowed_raises(self):
        al = RepoAllowlist(["myorg/*"])
        with pytest.raises(ValueError, match="not in ALLOWED_REPOS"):
            al.assert_allowed("evil", "repo")

    def test_assert_allowed_empty_list(self):
        al = RepoAllowlist([])
        with pytest.raises(ValueError, match="not in ALLOWED_REPOS"):
            al.assert_allowed("any", "repo")


class TestGetRepoAllowlist:
    def test_unset_returns_empty(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            al = get_repo_allowlist()
            assert al.patterns == []
            assert al.is_allowed("any", "repo") is False

    def test_single_pattern(self):
        with mock.patch.dict(os.environ, {"ALLOWED_REPOS": "myorg/*"}):
            al = get_repo_allowlist()
            assert al.is_allowed("myorg", "x") is True

    def test_comma_separated(self):
        with mock.patch.dict(os.environ, {"ALLOWED_REPOS": "orgA/*, orgB/repo"}):
            al = get_repo_allowlist()
            assert al.is_allowed("orgA", "anything") is True
            assert al.is_allowed("orgB", "repo") is True
            assert al.is_allowed("orgB", "other") is False

    def test_empty_string(self):
        with mock.patch.dict(os.environ, {"ALLOWED_REPOS": ""}):
            al = get_repo_allowlist()
            assert al.patterns == []
