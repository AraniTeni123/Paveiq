"""Tests for the paveiq package.

Kept deliberately light at project start; expand as stages are
implemented.
"""

from paveiq import __version__


def test_version_is_string():
    assert isinstance(__version__, str)
    assert __version__  # non-empty
