from jspace_demo import history

import pytest


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("JSPACE_HISTORY_DIR", str(tmp_path))
    return tmp_path


def test_save_list_load_and_delete(data_dir):
    first = history.save({"input": {"message": "a"}}, {"title": "a"})
    second = history.save({"input": {"message": "b"}}, {"title": "b"})
    assert [row["title"] for row in history.listing()] == ["b", "a"]  # newest first, even within one second
    record = history.load(first)
    assert record["input"] == {"message": "a"} and record["id"] == first and record["created"]
    assert history.delete(first) and history.load(first) is None and not history.delete(first)
    assert [row["id"] for row in history.listing()] == [second]
    assert not list(data_dir.glob("*.partial"))


def test_an_empty_history():
    assert history.listing() == []


@pytest.mark.parametrize("bad", ["../../etc/passwd", "20261006-120000-zzzzzz", "", "20261006-120000-abcdef.json"])
def test_only_an_id_reaches_the_disk(bad):
    assert history.load(bad) is None and history.delete(bad) is False
