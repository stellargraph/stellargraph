import pytest
import os
import subprocess


def test_black():
    p = subprocess.Popen(["black", "--check", "stellar", "tests"])
    out, err = p.communicate()
    assert p.returncode == 0
