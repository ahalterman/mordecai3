"""`mordecai3 index fetch`: verified download of the prebuilt index, and the
fall-back-to-building path when it cannot be had. Offline: the "mirrors" are
file:// URLs to a tiny archive."""

import hashlib
import io
import tarfile

import pytest
from typer.testing import CliRunner

from mordecai3 import index_builder as ib
from mordecai3.cli import app


def _archive(path, members=None):
    members = members or {"geonames_index/nodes/0/data.bin": b"segment"}
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    archive = src / ib.PREBUILT_NAME
    sha = _archive(archive)
    monkeypatch.setattr(ib, "PREBUILT_MIRRORS", [archive.as_uri()])
    monkeypatch.setattr(ib, "PREBUILT_SHA256", sha)
    monkeypatch.delenv("MORDECAI_INDEX_URL", raising=False)
    return archive


def test_fetch_verifies_and_unpacks(mirror, tmp_path):
    out = ib.fetch_prebuilt(tmp_path / "dest", progress=False)
    assert (out / "nodes/0/data.bin").read_bytes() == b"segment"
    assert not (tmp_path / "dest" / ib.PREBUILT_NAME).exists()   # archive removed


def test_checksum_mismatch_is_rejected_and_nothing_unpacked(mirror, tmp_path, monkeypatch):
    monkeypatch.setattr(ib, "PREBUILT_SHA256", "0" * 64)
    with pytest.raises(ib.FetchError) as e:
        ib.fetch_prebuilt(tmp_path / "dest", progress=False)
    assert "checksum mismatch" in str(e.value) and e.value.fallback_ok
    assert not (tmp_path / "dest" / ib.PREBUILT_TOP).exists()
    assert not list((tmp_path / "dest").glob("*.part"))


def test_falls_through_to_the_next_mirror(mirror, tmp_path, monkeypatch):
    monkeypatch.setattr(ib, "PREBUILT_MIRRORS",
                        [(tmp_path / "missing.tar.gz").as_uri(), mirror.as_uri()])
    assert ib.fetch_prebuilt(tmp_path / "dest", progress=False).is_dir()


def test_env_mirror_is_tried_first(mirror, tmp_path, monkeypatch):
    monkeypatch.setattr(ib, "PREBUILT_MIRRORS", [(tmp_path / "gone").as_uri()])
    monkeypatch.setenv("MORDECAI_INDEX_URL", mirror.as_uri())
    assert ib.fetch_prebuilt(tmp_path / "dest", progress=False).is_dir()


def test_archive_escaping_its_directory_is_refused(tmp_path, monkeypatch):
    bad = tmp_path / ib.PREBUILT_NAME
    sha = _archive(bad, {"geonames_index/../../evil": b"x"})
    monkeypatch.setattr(ib, "PREBUILT_MIRRORS", [bad.as_uri()])
    monkeypatch.setattr(ib, "PREBUILT_SHA256", sha)
    with pytest.raises(ib.FetchError, match="unexpected archive member"):
        ib.fetch_prebuilt(tmp_path / "dest", progress=False)
    assert not (tmp_path / "evil").exists()


def test_existing_index_dir_is_not_overwritten(mirror, tmp_path):
    (tmp_path / "dest" / ib.PREBUILT_TOP).mkdir(parents=True)
    (tmp_path / "dest" / ib.PREBUILT_TOP / "keep").write_text("mine")
    with pytest.raises(ib.FetchError) as e:
        ib.fetch_prebuilt(tmp_path / "dest", progress=False)
    assert not e.value.fallback_ok


def test_cli_failure_without_es_prints_the_build_route(tmp_path, monkeypatch):
    monkeypatch.setattr(ib, "PREBUILT_MIRRORS", [(tmp_path / "gone.tar.gz").as_uri()])
    monkeypatch.delenv("MORDECAI_INDEX_URL", raising=False)
    result = CliRunner().invoke(app, ["index", "fetch", "--dir", str(tmp_path),
                                      "--es-url", "http://127.0.0.1:1"])
    assert result.exit_code == 1
    assert "Could not get the prebuilt index" in result.output
    assert "mordecai3 index build" in result.output
    assert "docker run" in result.output
