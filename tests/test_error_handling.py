"""Tests for LSP error handling utilities."""

import pytest

from lean_lsp_mcp.utils import (
    LeanToolError,
    check_lsp_response,
)


class TestCheckLspResponse:
    """Tests for check_lsp_response function."""

    def test_returns_valid_response(self):
        """Valid responses should pass through unchanged."""
        response = {"result": "data"}
        assert check_lsp_response(response, "test_op") == response

    def test_returns_empty_list(self):
        """Empty list is a valid response (no results)."""
        response = []
        assert check_lsp_response(response, "test_op") == response

    def test_raises_on_none(self):
        """None response should raise LeanToolError (timeout)."""
        with pytest.raises(LeanToolError) as exc_info:
            check_lsp_response(None, "get_diagnostics")
        assert "LSP timeout during get_diagnostics" in str(exc_info.value)

    def test_raises_on_error_dict(self):
        """Error dict pattern should raise LeanToolError."""
        response = {"error": {"message": "Server crashed"}}
        with pytest.raises(LeanToolError) as exc_info:
            check_lsp_response(response, "get_hover")
        assert "LSP error during get_hover" in str(exc_info.value)
        assert "Server crashed" in str(exc_info.value)

    def test_raises_on_error_dict_without_message(self):
        """Error dict without message should use default."""
        response = {"error": {}}
        with pytest.raises(LeanToolError) as exc_info:
            check_lsp_response(response, "test_op")
        assert "unknown error" in str(exc_info.value)

    def test_allow_none_returns_none(self):
        """With allow_none=True, None should pass through."""
        assert check_lsp_response(None, "get_goal", allow_none=True) is None

    def test_allow_none_still_raises_on_error_dict(self):
        """With allow_none=True, error dict should still raise."""
        response = {"error": {"message": "Internal error"}}
        with pytest.raises(LeanToolError) as exc_info:
            check_lsp_response(response, "get_goal", allow_none=True)
        assert "LSP error during get_goal" in str(exc_info.value)
        assert "Internal error" in str(exc_info.value)

    def test_allow_none_returns_valid_response(self):
        """With allow_none=True, valid responses should pass through."""
        response = {"goals": ["some goal"]}
        assert check_lsp_response(response, "test_op", allow_none=True) == response


class TestLeanToolError:
    """Tests for LeanToolError exception."""

    def test_is_exception(self):
        """LeanToolError should be an Exception subclass."""
        assert issubclass(LeanToolError, Exception)

    def test_can_be_raised_with_message(self):
        """Should be raisable with a message."""
        with pytest.raises(LeanToolError) as exc_info:
            raise LeanToolError("Test error message")
        assert "Test error message" in str(exc_info.value)
