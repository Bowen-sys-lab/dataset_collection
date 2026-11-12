"""Shared utilities for deriving a replay trajectory without editing raw data."""
from pathlib import Path
import hashlib
import json


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def write_terminal_replay_source(out, original, records):
    """Write a derived sequence while preserving the original recording provenance."""
    original = Path(original).resolve()
    folder = Path(out) / 'source_sequence'
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / 'trajectory.jsonl'
    path.write_text(
        ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in records),
        encoding='utf-8',
    )
    metadata = json.loads((original.parent / 'run_metadata.json').read_text(encoding='utf-8'))
    metadata.update(
        record_count=len(records),
        trajectory_file=str(path.resolve()),
        source_trajectory_path=str(original),
        source_trajectory_sha256=_sha256(original),
        source_run_metadata_sha256=_sha256(original.parent / 'run_metadata.json'),
        derivation='exact recorded terminal snapshot; original label copied unchanged',
    )
    _write_json(folder / 'run_metadata.json', metadata)
    return path.resolve()
