import json

from darija_translator.jsonl import last_row_index, open_for_records, write_record

def test_write_record_keeps_arabic_script_readable(tmp_path):
    path = tmp_path / "pairs.jsonl"

    with open(path, "w", encoding="utf-8") as handle:
        write_record(handle, {"darija": "سلام"})

    assert "سلام" in path.read_text(encoding="utf-8")


def test_last_row_index_reads_where_a_run_stopped(tmp_path):
    path = tmp_path / "pairs.jsonl"

    with open(path, "w", encoding="utf-8") as handle:
        write_record(handle, {"row_index": 4})
        write_record(handle, {"row_index": 9})

    assert last_row_index(str(path)) == 9


def test_last_row_index_is_none_for_missing_or_empty_files(tmp_path):
    missing = tmp_path / "nope.jsonl"
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    assert last_row_index(str(missing)) is None
    assert last_row_index(str(empty)) is None


def test_written_records_round_trip_as_json(tmp_path):
    path = tmp_path / "pairs.jsonl"
    record = {"english": "Hello", "generated": "ahlan", "row_index": 1}

    with open(path, "w", encoding="utf-8") as handle:
        write_record(handle, record)

    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_open_for_records_truncates_when_not_resuming(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text("stale", encoding="utf-8")

    with open_for_records(str(path), resume=False) as handle:
        write_record(handle, {"row_index": 0})

    assert "stale" not in path.read_text(encoding="utf-8")


def test_open_for_records_appends_when_resuming(tmp_path):
    path = tmp_path / "pairs.jsonl"

    with open_for_records(str(path), resume=False) as handle:
        write_record(handle, {"row_index": 0})
    with open_for_records(str(path), resume=True) as handle:
        write_record(handle, {"row_index": 1})

    assert last_row_index(str(path)) == 1


def test_open_for_records_creates_the_directory(tmp_path):
    path = tmp_path / "nested" / "pairs.jsonl"

    with open_for_records(str(path), resume=False) as handle:
        write_record(handle, {"row_index": 0})

    assert path.exists()
