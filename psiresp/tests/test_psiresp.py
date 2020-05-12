"""
Unit and regression test for the psiresp package.
"""

# Import package, test suite, and other packages as needed
import psiresp
import pytest
import sys

def test_psiresp_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "psiresp" in sys.modules
