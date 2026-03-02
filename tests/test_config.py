"""Tests for stary.config – URL construction and validation helpers."""

import os
from unittest import mock

import pytest

from stary.config import build_dagster_run_url, get_dagster_base_url, validate_and_normalize_url


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
