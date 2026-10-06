from mordecai3.citation import maybe_show_citation_notice


def test_shown_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("MORDECAI_NO_CITATION_NOTICE", raising=False)
    maybe_show_citation_notice()
    assert "arXiv:2303.13675" in capsys.readouterr().err
    maybe_show_citation_notice()
    assert capsys.readouterr().err == ""


def test_silenced(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("MORDECAI_NO_CITATION_NOTICE", "1")
    maybe_show_citation_notice()
    assert capsys.readouterr().err == ""
    assert not (tmp_path / "mordecai3").exists()


def test_unwritable_marker_is_silent(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "mordecai3"
    blocker.write_text("")          # a file where the directory should go
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("MORDECAI_NO_CITATION_NOTICE", raising=False)
    maybe_show_citation_notice()
    assert capsys.readouterr().err == ""
