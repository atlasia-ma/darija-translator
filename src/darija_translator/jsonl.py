"""Streaming record IO, shared by the translate and DPO passes."""
import json
import os


def write_record(handle, record: dict) -> None:
    """One JSON object per line, unescaped so darija stays readable."""
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def last_row_index(path: str) -> int | None:
    """Where a previous interrupted run stopped, from the records it wrote."""
    if not os.path.exists(path):
        return None
    last = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    if last is None:
        return None
    return json.loads(last).get("row_index")


def open_for_records(path: str, resume: bool):
    """Append when resuming an existing file, otherwise start it fresh."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    mode = "a" if resume and os.path.exists(path) else "w"
    return open(path, mode, encoding="utf-8")
