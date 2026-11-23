"""Only the sealed, already existing artifact-root mkdir may be bypassed."""

from scripts import cscwd2027_review2_recovery_runner as adapter


def test_directory_adapter_is_narrow(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter.PARENT, "OUT", tmp_path)
    target = tmp_path / "artifacts"
    target.mkdir()
    called = []

    def original(path, **kwargs):
        called.append(path)

    adapter.allow_existing_root(original, target, exist_ok=False)
    assert not called
    adapter.allow_existing_root(original, tmp_path / "other", exist_ok=True)
    assert called == [tmp_path / "other"]
