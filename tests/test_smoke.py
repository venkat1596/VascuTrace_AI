"""Smoke test — proves the environment CI builds is the one we expect.

Replace/extend as VascuTrace_AI grows real modules; this exists so the CI gate
into main is live from the first commit rather than vacuously green.
"""

import sys


def test_python_is_3_13() -> None:
    assert sys.version_info[:2] == (3, 13)
