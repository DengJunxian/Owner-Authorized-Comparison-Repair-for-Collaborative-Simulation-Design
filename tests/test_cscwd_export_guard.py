"""The export scan distinguishes protocol IDs and regex source from secrets."""

import importlib.util
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[1]
    / "ccfa-workfiles/writing/paired-comparison/package_mechanism_review.py"
)
SPEC = importlib.util.spec_from_file_location("review_export_guard", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_task_identity_is_not_a_key():
    MODULE.scan("capsule.json", b'{"task_id":"task-' + b"a" * 64 + b'"}')


def test_regex_literal_is_not_private_key_material():
    MODULE.scan("guard.py", b're.compile(rb"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----")')


def test_secret_shaped_dummy_is_detected():
    with pytest.raises(RuntimeError):
        MODULE.scan("settings.json", b'{"dummy":"' + b"sk-" + b"a" * 32 + b'"}')


def test_local_user_path_is_detected():
    with pytest.raises(RuntimeError):
        MODULE.scan("record.json", b'{"path":"/Users/example/work/"}')
