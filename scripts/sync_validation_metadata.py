"""Metadata helpers for validated RGB/thermal capture synchronization."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
from copy import deepcopy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNC_EVIDENCE_PATH = Path(
    'validation/dual-camera-sync/spot-002-pixel-sync-recheck-20261013-b/summary.json'
)
SYNC_EVIDENCE_SHA256 = '9b5d87b6a9e6ce969f95a00237bfe592a98e4f105dee124333b5324ea01c6b73'
_EVIDENCE_CACHE: dict[str, dict[str, Any]] = {}
SYNC_CONTRACT_CHECKS = frozenset({
    'explicit_render_triggered', 'explicit_render_completed', 'frozen_state_capture',
    'camera_pose_match', 'simulation_time_frozen', 'paired_stream_resolutions',
    'thermal_geometry_complete', 'scene_config_matches_validation',
    'source_stage_matches_validation', 'thermal_config_matches_validation',
    'sensor_config_matches_validation', 'thermal_environment_matches_validation',
    'display_config_matches_validation', 'rgb_resolution_matches_validation',
    'thermal_resolution_matches_validation', 'rgb_camera_matches_validation',
    'thermal_camera_matches_validation', 'rgb_intrinsics_match_validation',
    'thermal_intrinsics_match_validation', 'renderer_matches_validation',
})


def frame_sync_validation_passes(
    frame: dict[str, Any], expected_evidence: dict[str, Any] | None = None,
) -> bool:
    """Validate the recorded per-frame contract rather than trusting one summary flag."""
    evidence = frame.get('sync_validation_evidence')
    checks = frame.get('sync_validation_checks')
    passed = (
        frame.get('strict_sync_verified') is True
        and frame.get('sync_status') == 'synchronized_pipeline_validated'
        and frame.get('sync_validation_status') == 'validated'
        and isinstance(checks, dict) and bool(checks)
        and SYNC_CONTRACT_CHECKS.issubset(checks)
        and all(value is True for value in checks.values())
        and isinstance(evidence, dict)
        and evidence.get('result') == 'PASS'
        and isinstance(evidence.get('report_sha256'), str)
    )
    if not passed or expected_evidence is None:
        return False
    return (
        evidence.get('report_sha256') == expected_evidence.get('report_sha256')
        and evidence.get('report') == expected_evidence.get('report')
    )


def frame_sync_contract_passes(frame: dict[str, Any]) -> bool:
    """True when this frame's own capture contract passed, whatever its sync basis.

    Strict probe inheritance and project-convention synchronisation are different
    claims; both require the frame-level checks (time frozen, pose match, render
    triggered, coverage, paired resolutions) to have actually passed.
    """
    checks = frame.get('sync_validation_checks')
    return isinstance(checks, dict) and bool(checks) and all(value is True for value in checks.values())


def _evidence_available(project_root: Path = PROJECT_ROOT) -> tuple[dict[str, Any] | None, str | None]:
    """Load the independent probe evidence when it exists.

    A *missing* evidence file is not an error any more: a scene that was never
    covered by the school marker probe is synchronised by project convention
    instead. A *present but altered* file still raises, because that is an
    integrity problem rather than a scope problem.
    """
    path = Path(project_root) / SYNC_EVIDENCE_PATH
    if not path.is_file():
        return None, f'sync validation evidence is unavailable: {path}'
    return _load_verified_evidence(project_root), None


def _load_verified_evidence(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    path = Path(project_root) / SYNC_EVIDENCE_PATH
    cache_key = str(path.resolve())
    if cache_key in _EVIDENCE_CACHE:
        return deepcopy(_EVIDENCE_CACHE[cache_key])
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f'sync validation evidence is unavailable: {path}') from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SYNC_EVIDENCE_SHA256:
        raise RuntimeError(f'sync validation evidence hash mismatch: {path}')
    report = json.loads(raw)
    frames = report.get('frames')
    if (report.get('result') != 'PASS'
            or report.get('strict_sync_verified') is not True
            or report.get('strict_per_frame_pixel_sync_verified') is not True
            or not isinstance(frames, list) or len(frames) != 16
            or report.get('marker_history_verified') is not True
            or report.get('marker_history_max_age_frames') != 0
            or report.get('failed_frames')
            or not all(isinstance(frame, dict)
                       and frame.get('pass') is True
                       and frame.get('strict_sync_verified') is True
                       and frame.get('strict_per_frame_pixel_sync_verified') is True
                       for frame in frames)):
        raise RuntimeError(f'sync validation evidence does not meet the expected pass criteria: {path}')
    evidence = {
        'report': SYNC_EVIDENCE_PATH.as_posix(),
        'report_sha256': digest,
        'result': report['result'],
        'frames_tested': len(frames),
        'strict_per_frame_pixel_sync_verified': True,
        'marker_history_verified': True,
        'marker_history_max_age_frames': 0,
        'scene_config_path': report.get('scene_config_path'),
        'scene_config_sha256': report.get('scene_config_sha256'),
        'trajectory': report.get('trajectory'),
        'trajectory_sha256': report.get('trajectory_sha256'),
        'rgb_resolution': report.get('rgb_resolution'),
        'thermal_resolution': report.get('thermal_resolution'),
        'source_stage_sha256': report.get('source_usd_sha256_before'),
        'scene_config_sha256': report.get('scene_config_sha256'),
        'renderer': report.get('renderer'),
    }
    conditions = report.get('experiment_conditions', {})
    files = conditions.get('files', {})
    for name in ('thermal_config', 'sensor_config'):
        record = files.get(name)
        evidence[f'{name}_sha256'] = record.get('sha256') if isinstance(record, dict) else None
    for name in ('thermal_environment_config', 'display_config'):
        record = files.get(name)
        evidence[f'{name}_sha256'] = record.get('sha256') if isinstance(record, dict) else None
    first_frame = frames[0]
    evidence['rgb_camera_path'] = first_frame.get('rgb_camera_path')
    evidence['thermal_camera_path'] = first_frame.get('thermal_camera_path')
    evidence['rgb_intrinsics_px'] = first_frame.get('rgb_intrinsics_px')
    evidence['thermal_intrinsics_px'] = first_frame.get('thermal_intrinsics_px')
    required_evidence = (
        evidence['source_stage_sha256'], evidence['scene_config_sha256'],
        evidence['thermal_config_sha256'], evidence['sensor_config_sha256'],
        evidence['thermal_environment_config_sha256'], evidence['display_config_sha256'],
        evidence['renderer'], evidence['rgb_camera_path'], evidence['thermal_camera_path'],
        evidence['rgb_resolution'], evidence['thermal_resolution'],
        evidence['rgb_intrinsics_px'], evidence['thermal_intrinsics_px'],
    )
    if any(value is None for value in required_evidence):
        raise RuntimeError(f'sync validation evidence is missing configuration scope: {path}')
    _EVIDENCE_CACHE[cache_key] = evidence
    return deepcopy(evidence)


def build_sync_validation_metadata(
    frame_metadata: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Return honest sync labels for this capture mode and its runtime checks.

    Offline frames inherit a validated capture-pipeline probe and must also pass
    their own frozen-state, render, camera-pose, time, and coverage checks.
    Pixel-level marker verification remains explicitly per-run evidence.
    """
    capture_mode = frame_metadata.get('capture_mode')
    if capture_mode == 'live_preview':
        return {
            'sync_basis': 'live_preview',
            'strict_sync_verified': False,
            'strict_per_frame_pixel_sync_verified': False,
            'sync_status': 'live_preview_not_strict',
            'sync_validation_status': 'not_verified',
            'sync_validation_scope': 'live_preview',
            'sync_validation_evidence': None,
            'sync_validation_checks': {},
            'sync_note': 'Live preview is asynchronous and is not labeled as verified paired capture.',
        }
    if capture_mode != 'offline_frozen_state_render':
        return {
            'sync_basis': 'unsupported_capture_mode',
            'strict_sync_verified': False,
            'strict_per_frame_pixel_sync_verified': False,
            'sync_status': 'not_verified',
            'sync_validation_status': 'not_verified',
            'sync_validation_scope': 'unsupported_capture_mode',
            'sync_validation_evidence': None,
            'sync_validation_checks': {},
            'sync_note': f'Capture mode {capture_mode!r} is outside the validated offline pipeline.',
        }

    try:
        pose_error = float(frame_metadata['pair_pose_max_error'])
        capture_drift = float(frame_metadata['simulation_capture_drift_s'])
        before = float(frame_metadata['simulation_time_before'])
        after = float(frame_metadata['simulation_time_after'])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'sync contract is missing valid time/pose diagnostics: {exc}') from exc
    try:
        rgb_width, rgb_height = (int(value) for value in frame_metadata['rgb_resolution'])
        thermal_width, thermal_height = (int(value) for value in frame_metadata['thermal_resolution'])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'sync contract is missing valid stream resolutions: {exc}') from exc
    dimensions_valid = min(rgb_width, rgb_height, thermal_width, thermal_height) > 0
    paired_aspect = (dimensions_valid and
                     abs(rgb_width / rgb_height - thermal_width / thermal_height)
                     / max(rgb_width / rgb_height, thermal_width / thermal_height) <= 0.005)
    checks = {
        'explicit_render_triggered': frame_metadata.get('explicit_render_triggered') is True,
        'explicit_render_completed': frame_metadata.get('explicit_render_completed') is True,
        'frozen_state_capture': frame_metadata.get('frozen_state_capture') is True,
        'camera_pose_match': math.isfinite(pose_error) and pose_error <= 1e-6,
        'simulation_time_frozen': math.isfinite(before) and math.isfinite(after)
        and abs(after - before) <= 1e-8
        and math.isfinite(capture_drift) and abs(capture_drift) <= 1e-8,
        'paired_stream_resolutions': paired_aspect,
        'thermal_geometry_complete': frame_metadata.get('thermal_coverage', {}).get(
            'complete_for_this_view') is True,
    }
    if not all(checks.values()):
        failures = ', '.join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f'sync contract failed: {failures}')

    evidence, unavailable_reason = _evidence_available(project_root)
    scope_checks = {} if evidence is None else {
        'scene_config_matches_validation': frame_metadata.get('scene_config_sha256') == evidence['scene_config_sha256'],
        'source_stage_matches_validation': frame_metadata.get('source_stage_sha256') == evidence['source_stage_sha256'],
        'thermal_config_matches_validation': frame_metadata.get('thermal_config_sha256') == evidence['thermal_config_sha256'],
        'sensor_config_matches_validation': frame_metadata.get('sensor_config_sha256') == evidence['sensor_config_sha256'],
        'thermal_environment_matches_validation': frame_metadata.get('thermal_environment_config_sha256') == evidence['thermal_environment_config_sha256'],
        'display_config_matches_validation': frame_metadata.get('display_config_sha256') == evidence['display_config_sha256'],
        'rgb_resolution_matches_validation': frame_metadata.get('rgb_resolution') == evidence['rgb_resolution'],
        'thermal_resolution_matches_validation': frame_metadata.get('thermal_resolution') == evidence['thermal_resolution'],
        'rgb_camera_matches_validation': frame_metadata.get('rgb_camera_path') == evidence['rgb_camera_path'],
        'thermal_camera_matches_validation': frame_metadata.get('thermal_camera_path') == evidence['thermal_camera_path'],
        'rgb_intrinsics_match_validation': frame_metadata.get('rgb_intrinsics_px') == evidence['rgb_intrinsics_px'],
        'thermal_intrinsics_match_validation': frame_metadata.get('thermal_intrinsics_px') == evidence['thermal_intrinsics_px'],
        'renderer_matches_validation': frame_metadata.get('rgb_render_options', {}).get('renderer') == evidence['renderer'],
    }
    if evidence is not None and all(scope_checks.values()):
        checks.update(scope_checks)
        return {
            'sync_basis': 'strict_probe',
            'strict_sync_verified': True,
            'strict_per_frame_pixel_sync_verified': False,
            'sync_status': 'synchronized_pipeline_validated',
            'sync_validation_status': 'validated',
            'sync_validation_scope': 'pipeline_and_frame_capture_contract',
            'sync_validation_evidence': evidence,
            'sync_validation_checks': checks,
            'sync_scope_checks': scope_checks,
            'sync_scope_failures': [],
            'sync_note': (
                'The offline capture pipeline passed independent moving-marker validation, and this frame '
                'passed its frozen-state/render/pose/time/thermal-coverage contract. Per-frame marker pixel '
                'verification is reported separately and was not rerun for this frame.'
            ),
        }
    # This scene is not the one the school marker probe covered, so the strict claim
    # cannot be inherited. The frame still passed its own capture contract, so the
    # data is usable as project-convention synchronised data -- with the strict
    # fields deliberately left false rather than quietly promoted.
    scope_failures = [name for name, passed in scope_checks.items() if not passed]
    return {
        'sync_basis': 'project_convention',
        'strict_sync_verified': False,
        'strict_per_frame_pixel_sync_verified': False,
        'sync_status': 'synchronized_by_project_convention',
        'sync_validation_status': 'convention_only',
        'sync_validation_scope': 'frame_capture_contract_only',
        'sync_validation_evidence': None,
        'sync_validation_checks': checks,
        'sync_scope_checks': scope_checks,
        'sync_scope_failures': scope_failures,
        'sync_validation_evidence_unavailable': unavailable_reason,
        'sync_note': (
            'This frame passed its own frozen-state/render/pose/time/thermal-coverage contract. '
            'The independent moving-marker probe does not cover this scene, configuration or resolution, '
            'so synchronisation is claimed by project convention only; per-frame pixel verification was '
            'not performed and is not implied.'
        ),
    }
