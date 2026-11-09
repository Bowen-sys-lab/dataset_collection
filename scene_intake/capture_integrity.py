"""Read-only verification of a sealed forest capture; never renew its hashes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(capture, *, required=True):
    capture=Path(capture).resolve()
    if (capture/'failure.txt').exists():raise ValueError('capture is marked failed')
    path=capture/'collection_manifest.json'
    if not path.is_file():
        if required:raise ValueError('sealed capture manifest is required')
        return None
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(manifest,dict) or manifest.get('schema_version')!=1 or not isinstance(manifest.get('files'),dict) or not manifest['files']:
        raise ValueError('invalid capture manifest')
    actual={p.relative_to(capture).as_posix() for p in capture.rglob('*') if p.is_file() and p!=path}
    if actual!=set(manifest['files']):raise ValueError('capture manifest does not cover the complete output inventory')
    for name,digest in manifest['files'].items():
        item=(capture/name).resolve()
        if not item.is_relative_to(capture) or not item.is_file() or file_sha256(item)!=digest:
            raise ValueError(f'capture integrity mismatch: {name}')
    return manifest


def verify_video_input(capture):
    """Enforce optional seals at the existing video consumer, including its source."""
    capture=Path(capture).resolve()
    verify_manifest(capture,required=False)
    path=capture/'video_input_manifest.json'
    if not path.is_file():return
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(manifest,dict) or manifest.get('schema_version')!=1 or not isinstance(manifest.get('files'),dict) or not manifest['files']:
        raise ValueError('invalid video input manifest')
    actual={p.relative_to(capture).as_posix() for p in capture.rglob('*') if p.is_file() and p!=path}
    if actual!=set(manifest['files']):raise ValueError('video input manifest inventory mismatch')
    for name,digest in manifest['files'].items():
        item=(capture/name).resolve()
        if not item.is_relative_to(capture) or not item.is_file() or file_sha256(item)!=digest:
            raise ValueError(f'video input integrity mismatch: {name}')
    source_value=manifest.get('source_capture')
    if not isinstance(source_value,str) or not Path(source_value).is_absolute():
        raise ValueError('video source capture path is required')
    source=Path(source_value).resolve()
    if source==capture:raise ValueError('video source must be a separate sealed capture')
    original=verify_manifest(source)
    if file_sha256(source/'collection_manifest.json')!=manifest.get('source_manifest_sha256'):
        raise ValueError('video source manifest identity mismatch')
    for name,digest in manifest['files'].items():
        if name!='replay_metadata.json' and original['files'].get(name)!=digest:
            raise ValueError(f'video input differs from its sealed source: {name}')
