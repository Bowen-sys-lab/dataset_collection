"""Create a provenance-preserving side-by-side RGB/thermal replay video.

This script is deliberately Isaac-independent.  It consumes an existing
offline replay and never re-renders or alters the source RGB, thermal, numpy,
or per-frame metadata files.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import math
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.trajectory_recorder import trajectory_provenance
from scripts.frame_metadata import FrameMetadataReader
from scripts.sync_validation_metadata import _load_verified_evidence, frame_sync_validation_passes
DEFAULT_FFMPEG_ROOT = PROJECT_ROOT / "tools" / "ffmpeg"
EPSILON = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: expected a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field}: expected a finite number")
    return result


def _same_sequence(left: Any, right: Any, tolerance: float = EPSILON) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return left == right
    if len(left) != len(right):
        return False
    return all(_same_sequence(a, b, tolerance) if isinstance(a, (list, tuple))
               else abs(_number(a, "sequence") - _number(b, "sequence")) <= tolerance
               for a, b in zip(left, right))


@dataclass(frozen=True)
class ReplayFrame:
    index: int
    trajectory_frame_id: int
    trajectory_index: int
    replay_trajectory_index: int
    simulation_time_s: float
    video_time_s: float
    duration_s: float
    position: Any
    rgb_path: Path
    thermal_path: Path
    metadata_path: Path
    strict_sync_verified: bool
    sync_contract_verified: bool


@dataclass(frozen=True)
class ReplayBundle:
    root: Path
    metadata: dict[str, Any]
    frames: tuple[ReplayFrame, ...]
    rgb_width: int
    rgb_height: int
    thermal_width: int
    thermal_height: int
    strict_sync_verified: bool
    trajectory_path: Path
    run_metadata: dict[str, Any]

    @property
    def width(self) -> int:
        """Backward-compatible alias for the RGB presentation-panel width."""
        return self.rgb_width

    @property
    def height(self) -> int:
        """Backward-compatible alias for the RGB presentation-panel height."""
        return self.rgb_height


def _trajectory_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"trajectory file is not readable: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"trajectory line {line_number}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"trajectory line {line_number}: expected an object")
        records.append(record)
    if not records:
        raise ValueError(f"trajectory is empty: {path}")
    return records


def load_replay(replay: str | Path) -> ReplayBundle:
    """Load and validate one replay without opening Isaac Sim.

    Validation intentionally rejects mismatched trajectory metadata.  A video
    must never silently pair an image with a pose from a different run.
    """
    root = Path(replay).resolve()
    from scene_intake.capture_integrity import verify_video_input
    verify_video_input(root)
    metadata_path = root / "replay_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"missing replay metadata: {metadata_path}")
    metadata = _read_json(metadata_path)
    if metadata.get("result") != "PASS" or (root / "failure.txt").is_file():
        raise ValueError("video generation requires a successful completed replay")
    try:
        frame_count = int(metadata["frames"])
        legacy_resolution = metadata["resolution"]
        rgb_resolution = metadata.get("rgb_resolution", legacy_resolution)
        thermal_resolution = metadata.get("thermal_resolution", legacy_resolution)
        width, height = (int(rgb_resolution[0]), int(rgb_resolution[1]))
        thermal_width, thermal_height = (int(thermal_resolution[0]), int(thermal_resolution[1]))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("replay metadata must contain frames and valid stream resolutions") from exc
    if frame_count <= 0 or min(width, height, thermal_width, thermal_height) <= 0:
        raise ValueError("replay frames and stream resolutions must be positive")
    trajectory_path = Path(metadata.get("trajectory", ""))
    if not trajectory_path.is_absolute():
        trajectory_path = root / trajectory_path
    if not trajectory_path.is_file():
        raise ValueError(f"trajectory file is not readable: {trajectory_path}")
    provenance = trajectory_provenance(trajectory_path)
    recorded_provenance = metadata.get('trajectory_provenance')
    # New versions add provenance fields. Compare every field actually saved
    # by the older replay, while keeping the source metadata hash immutable.
    if recorded_provenance is not None and any(
            provenance.get(key) != value for key, value in recorded_provenance.items()):
        raise ValueError('trajectory/task provenance does not match the recorded replay')
    recorded_stage_hash = provenance.get('recording_stage_sha256')
    replay_stage_hash = metadata.get('source_stage_sha256')
    if recorded_stage_hash is not None and replay_stage_hash is not None and recorded_stage_hash != replay_stage_hash:
        raise ValueError('The recorded source USD differs from the replay USD')
    metadata['trajectory_provenance'] = provenance
    trajectory = _trajectory_records(trajectory_path)
    by_id = {record.get("frame_id"): record for record in trajectory}
    if len(by_id) != len(trajectory) or any(type(key) is not int for key in by_id):
        raise ValueError("trajectory frame_id values must be unique")
    trajectory_rows = {record["frame_id"]: index for index, record in enumerate(trajectory)}

    json_frame_paths = sorted(root.glob("frame_*.json"))
    if len(json_frame_paths) != frame_count:
        raise ValueError(f"replay frame metadata count {len(json_frame_paths)} does not match {frame_count}")
    for kind in ("rgb", "thermal"):
        if len(list(root.glob(f"frame_*_{kind}.png"))) != frame_count:
            raise ValueError(f"replay {kind} image count does not match {frame_count}; missing or extra file")
    frames: list[ReplayFrame] = []
    frame_metadata_reader = FrameMetadataReader()
    first_time = _number(frame_metadata_reader.read(root / "frame_000000.json").get(
        "replay_simulation_time"), "first replay time")
    previous_time: float | None = None
    for index in range(frame_count):
        stem = f"frame_{index:06d}"
        frame_meta_path = root / f"{stem}.json"
        rgb_path = root / f"{stem}_rgb.png"
        thermal_path = root / f"{stem}_thermal.png"
        for path in (frame_meta_path, rgb_path, thermal_path):
            if not path.is_file():
                raise ValueError(f"missing replay file: {path.name}")
        frame_meta = frame_metadata_reader.read(frame_meta_path)
        if frame_meta.get("frame_id") != index:
            raise ValueError(f"{stem}: frame_id mismatch")
        if recorded_provenance is not None:
            if frame_meta.get('task_id') != (provenance['task'] or {}).get('task_id') or \
                    frame_meta.get('task_result_status') != provenance['task_result']['status'] or \
                    frame_meta.get('trajectory_sha256') != provenance['trajectory_sha256']:
                raise ValueError(f'{stem}: task/trajectory provenance mismatch')
        frame_rgb_resolution = frame_meta.get("rgb_resolution", frame_meta.get("resolution", [width, height]))
        frame_thermal_resolution = frame_meta.get(
            "thermal_resolution", frame_meta.get("resolution", [thermal_width, thermal_height]))
        try:
            frame_rgb_resolution = [int(frame_rgb_resolution[0]), int(frame_rgb_resolution[1])]
            frame_thermal_resolution = [int(frame_thermal_resolution[0]), int(frame_thermal_resolution[1])]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"{stem}: invalid per-stream resolution metadata") from exc
        if frame_rgb_resolution != [width, height] or frame_thermal_resolution != [thermal_width, thermal_height]:
            raise ValueError(f"{stem}: frame stream resolution does not match replay metadata")
        trajectory_frame_id = frame_meta.get("trajectory_frame_id")
        record = by_id.get(trajectory_frame_id)
        if record is None:
            raise ValueError(f"{stem}: trajectory frame_id {trajectory_frame_id!r} is missing")
        replay_time = _number(frame_meta.get("replay_simulation_time"), f"{stem} replay time")
        trajectory_time = _number(record.get("simulation_time"), "trajectory time")
        if abs(replay_time - trajectory_time) > EPSILON:
            raise ValueError(f"{stem}: trajectory time does not match replay time")
        source_pose = frame_meta.get('source_trajectory_pose')
        if not isinstance(source_pose, dict):
            source_pose = frame_meta
        if "position" in source_pose and "position" in record and not _same_sequence(
                source_pose["position"], record["position"]):
            raise ValueError(f"{stem}: trajectory position does not match recorded source pose")
        if "orientation" in source_pose and "orientation" in record:
            q1, q2 = source_pose["orientation"], record["orientation"]
            if not _same_sequence(q1, q2) and not _same_sequence(q1, [-float(v) for v in q2]):
                raise ValueError(f"{stem}: trajectory orientation does not match recorded source pose")
        if previous_time is not None and replay_time <= previous_time:
            raise ValueError(f"{stem}: replay timestamps must be strictly increasing")
        previous_time = replay_time
        with Image.open(rgb_path) as rgb, Image.open(thermal_path) as thermal:
            if rgb.size != (width, height) or thermal.size != (thermal_width, thermal_height):
                raise ValueError(f"{stem}: image resolution does not match replay metadata")
        frames.append(ReplayFrame(
            index=index,
            trajectory_frame_id=int(trajectory_frame_id),
            trajectory_index=trajectory_rows[trajectory_frame_id],
            replay_trajectory_index=int(frame_meta.get("trajectory_index", index)),
            simulation_time_s=replay_time,
            video_time_s=round(replay_time - first_time, 9),
            duration_s=0.0,
            position=frame_meta.get("position"),
            rgb_path=rgb_path,
            thermal_path=thermal_path,
            metadata_path=frame_meta_path,
            strict_sync_verified=bool(frame_meta.get("strict_per_frame_pixel_sync_verified", False)),
            sync_contract_verified=frame_sync_validation_passes(
                frame_meta, metadata.get('sync_validation_evidence')),
        ))
    durations = [
        round(frames[i + 1].simulation_time_s - frames[i].simulation_time_s, 9)
        if i + 1 < len(frames) else _number(metadata.get("sample_interval_s", 0.2), "final frame hold")
        for i in range(len(frames))
    ]
    if any(duration <= 0 for duration in durations):
        raise ValueError("replay timestamps produce a non-positive video duration")
    frames = [replace(frame, duration_s=duration)
              for frame, duration in zip(frames, durations)]
    # Preserve a validated source capture contract, while keeping the stronger
    # per-frame pixel-marker result separate on each video row.
    sync_evidence = metadata.get('sync_validation_evidence')
    try:
        current_sync_evidence = _load_verified_evidence()
    except RuntimeError:
        current_sync_evidence = None
    strict = bool(
        metadata.get('strict_sync_verified') is True
        and metadata.get('sync_validation_status') == 'validated'
        and isinstance(sync_evidence, dict)
        and sync_evidence.get('result') == 'PASS'
        and current_sync_evidence is not None
        and sync_evidence.get('report') == current_sync_evidence.get('report')
        and sync_evidence.get('report_sha256') == current_sync_evidence.get('report_sha256')
        and all(frame.sync_contract_verified for frame in frames)
    )
    run_path = trajectory_path.parent / "run_metadata.json"
    run_metadata = _read_json(run_path) if run_path.is_file() else {}
    return ReplayBundle(root, metadata, tuple(frames), width, height,
                        thermal_width, thermal_height, strict,
                        trajectory_path, run_metadata)


def _font(size=22) -> ImageFont.ImageFont:
    candidates = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "segoeui.ttf",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def compose_frame(bundle: ReplayBundle, index: int) -> tuple[Image.Image, dict[str, tuple[int, int, int, int]]]:
    frame = bundle.frames[index]
    with Image.open(frame.rgb_path) as rgb_source, Image.open(frame.thermal_path) as thermal_source:
        rgb = rgb_source.convert("RGB")
        thermal = thermal_source.convert("RGB")
    thermal_fit = ImageOps.contain(thermal, (bundle.width, bundle.height),
                                   method=Image.Resampling.NEAREST)
    thermal_panel = Image.new("RGB", (bundle.width, bundle.height), (0, 0, 0))
    thermal_panel.paste(thermal_fit, ((bundle.width - thermal_fit.width) // 2,
                                      (bundle.height - thermal_fit.height) // 2))
    bar_height = 96
    canvas_width = max(bundle.width * 2, 1000)
    image = Image.new("RGB", (canvas_width + canvas_width % 2,
                              bundle.height + 208 + bundle.height % 2), (22, 24, 28))
    image.paste(rgb, (0, bar_height))
    image.paste(thermal_panel, (bundle.width, bar_height))
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((16, 10), "RGB", fill=(240, 240, 240), font=font)
    draw.text((bundle.width + 16, 10), "THERMAL", fill=(240, 240, 240), font=font)
    if bundle.metadata.get('confirmation_mode')=='manual_only':
        draw.text((16,73),'MANUAL CONFIRMATION PREVIEW | no automatic task qualification',
                  fill=(255,201,86),font=_font(16))
    draw.text((16, 43),
              f"video_t={frame.video_time_s:.3f}s | sim_t={frame.simulation_time_s:.3f}s | "
              f"trajectory_frame={frame.trajectory_frame_id} | index={frame.index}",
              fill=(205, 210, 216), font=font)
    footer_y = bar_height + bundle.height
    draw.line((0, footer_y, image.width, footer_y), fill=(70, 80, 90))
    # This is a top-down XY trajectory inset, not a scene segmentation map.
    valid_positions = [f.position for f in bundle.frames if f.position is not None]
    if valid_positions:
        min_x = min(float(p[0]) for p in valid_positions)
        min_y = min(float(p[1]) for p in valid_positions)
        span_x = max(float(p[0]) for p in valid_positions) - min_x
        span_y = max(float(p[1]) for p in valid_positions) - min_y
        scale = min(184 / max(span_x, 1e-6), 76 / max(span_y, 1e-6))
        route = [(18 + (float(f.position[0]) - min_x) * scale,
                  footer_y + 94 - (float(f.position[1]) - min_y) * scale)
                 for f in bundle.frames if f.position is not None]
        if len(route) > 1:
            draw.line(route, fill=(75, 87, 99), width=2)
        current_route = route[:index + 1]
        if len(current_route) > 1:
            draw.line(current_route, fill=(90, 205, 230), width=3)
        if current_route:
            x, y = current_route[-1]
            draw.ellipse((x-4, y-4, x+4, y+4), fill=(255, 201, 86))
        draw.text((16, footer_y + 2), "XY 轨迹（俯视）", fill=(205, 210, 216), font=_font(16))
    if frame.position is not None:
        xyz = ", ".join(f"{float(v):.2f}" for v in frame.position)
        draw.text((224, footer_y + 40), f"世界坐标 XYZ（米）：{xyz}",
                  fill=(160, 175, 190), font=_font(18))
    return image, {
        "rgb": (0, bar_height, bundle.width, bar_height + bundle.height),
        "thermal": (bundle.width, bar_height, bundle.width * 2, bar_height + bundle.height),
    }


def find_ffmpeg(path: str | Path | None) -> Path:
    if path:
        candidate = Path(path).resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"FFmpeg executable not found: {candidate}")
    candidates = sorted(DEFAULT_FFMPEG_ROOT.glob("**/bin/ffmpeg.exe"))
    if candidates:
        return candidates[-1].resolve()
    system = shutil.which("ffmpeg")
    if system:
        return Path(system).resolve()
    raise FileNotFoundError(
        "FFmpeg was not found. Put a portable ffmpeg.exe under tools/ffmpeg or pass --ffmpeg.")


def _ffprobe_for(ffmpeg: Path) -> Path | None:
    sibling = ffmpeg.with_name("ffprobe" + ffmpeg.suffix)
    if sibling.is_file():
        return sibling
    found = shutil.which("ffprobe")
    return Path(found).resolve() if found else None


def _probe_frames(ffmpeg: Path, video: Path) -> tuple[int, list[float]]:
    ffprobe = _ffprobe_for(ffmpeg)
    if ffprobe is None:
        raise FileNotFoundError("ffprobe is required to verify encoded frame count and timestamps")
    command = [str(ffprobe), "-v", "error", "-select_streams", "v:0", "-count_frames",
               "-show_entries", "stream=nb_read_frames", "-show_entries", "frame=best_effort_timestamp_time",
               "-of", "json", str(video)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout or "{}")
    stream = (payload.get("streams") or [{}])[0]
    timestamps = [float(frame["best_effort_timestamp_time"])
                  for frame in payload.get("frames", []) if "best_effort_timestamp_time" in frame]
    return int(stream.get("nb_read_frames") or len(timestamps)), timestamps


def generate_video(replay: str | Path, out: str | Path, *, ffmpeg: str | Path | None = None
                   ) -> dict[str, Any]:
    bundle = load_replay(replay)
    output = Path(out).resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    encoder = find_ffmpeg(ffmpeg)
    if _ffprobe_for(encoder) is None:
        raise FileNotFoundError("ffprobe is required next to ffmpeg or on PATH")
    output.mkdir(parents=True)
    source_files = [bundle.root / "replay_metadata.json", bundle.trajectory_path]
    source_files.extend(path for frame in bundle.frames for path in (
        frame.rgb_path, frame.thermal_path, frame.metadata_path))
    common_metadata_path = bundle.root / 'common_metadata.json'
    if common_metadata_path.is_file():
        source_files.append(common_metadata_path)
    run_metadata_path = bundle.trajectory_path.parent / "run_metadata.json"
    if run_metadata_path.is_file():
        source_files.append(run_metadata_path)
    source_hashes_before = {str(path): _sha256(path) for path in source_files}
    with tempfile.TemporaryDirectory(prefix="replay-video-", dir=str(output.parent)) as temp_name:
        temp = Path(temp_name)
        concat = temp / "frames.txt"
        concat_lines: list[str] = []
        frame_rows = []
        for index, frame in enumerate(bundle.frames):
            composed, _ = compose_frame(bundle, index)
            png = temp / f"video_{index:06d}.png"
            composed.save(png, format="PNG", compress_level=1)
            # concat demuxer paths are single-quoted and use doubled quotes.
            escaped = str(png).replace("'", "'\\''")
            concat_lines.extend([f"file '{escaped}'", "option framerate 1000000",
                                 f"duration {frame.duration_s:.9f}"])
            if index in (0, len(bundle.frames) // 2, len(bundle.frames) - 1):
                composed.save(output / f"preview_{index:06d}.png")
            frame_rows.append({
                "video_frame_index": index,
                "trajectory_frame_id": frame.trajectory_frame_id,
                "trajectory_index": frame.trajectory_index,
                "replay_trajectory_index": frame.replay_trajectory_index,
                "simulation_time_s": frame.simulation_time_s,
                "video_time_s": frame.video_time_s,
                "duration_s": frame.duration_s,
                "rgb_source": frame.rgb_path.name,
                "thermal_source": frame.thermal_path.name,
                "rgb_resolution": [bundle.width, bundle.height],
                "thermal_resolution": [bundle.thermal_width, bundle.thermal_height],
                "thermal_video_resize": "nearest-neighbor fit to RGB panel; source image unchanged",
                "strict_sync_verified": frame.sync_contract_verified,
                "strict_per_frame_pixel_sync_verified": frame.strict_sync_verified,
                "sync_basis": bundle.metadata.get('sync_basis', 'not_verified'),
                "sync_status": bundle.metadata.get('sync_status', 'not_verified'),
                "sync_validation_status": bundle.metadata.get('sync_validation_status', 'not_verified'),
                "sync_validation_scope": bundle.metadata.get('sync_validation_scope'),
                "data_usage": "pilot/development data",
                "task_id": (bundle.metadata['trajectory_provenance']['task'] or {}).get('task_id'),
                "task_result_status": bundle.metadata['trajectory_provenance']['task_result']['status'],
            })
        # concat repeats the final file so its duration is honored by ffmpeg.
        concat_lines.extend([f"file '{escaped}'", "option framerate 1000000"])
        concat.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        video_path = output / "paired_replay.mp4"
        command = [str(encoder), "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "concat", "-safe", "0", "-i", str(concat),
                   "-fps_mode", "passthrough",
                   "-enc_time_base", "1:1000000", "-frames:v", str(len(bundle.frames)),
                   "-c:v", "libx264", "-preset", "medium", "-bf", "0",
                   "-bsf:v", f"setts=duration='if(eq(N,{len(bundle.frames)-1}),"
                   f"{round(bundle.frames[-1].duration_s * 1000000)},DURATION)'",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   "-video_track_timescale", "1000000",
                   str(video_path)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            (output / "ffmpeg_failure.txt").write_text(result.stderr, encoding="utf-8")
            raise RuntimeError(f"FFmpeg failed with exit code {result.returncode}; "
                               f"see ffmpeg_failure.txt\n{result.stderr[-2000:]}")
    source_files_unchanged = source_hashes_before == {str(path): _sha256(path) for path in source_files}
    from scene_intake.capture_integrity import verify_video_input
    verify_video_input(bundle.root)
    encoded_count, encoded_pts = _probe_frames(encoder, video_path)
    if encoded_count != len(bundle.frames):
        raise RuntimeError(f"encoded frame count {encoded_count} != replay frame count {len(bundle.frames)}")
    if len(encoded_pts) != len(bundle.frames) or any(
            abs(t - f.video_time_s) > 2e-6 for t, f in zip(encoded_pts, bundle.frames)):
        raise RuntimeError("encoded timestamps do not match replay-relative simulation times")
    if not source_files_unchanged:
        raise RuntimeError("source data changed during video generation")
    duration_probe = subprocess.run(
        [str(_ffprobe_for(encoder)), "-v", "error", "-show_entries", "stream=duration,width,height",
         "-of", "json", str(video_path)], capture_output=True, text=True, check=True)
    stream = json.loads(duration_probe.stdout)["streams"][0]
    encoded_duration = float(stream["duration"])
    expected_duration = bundle.frames[-1].video_time_s + bundle.frames[-1].duration_s
    if abs(encoded_duration - expected_duration) > 2e-6:
        raise RuntimeError(f"encoded duration {encoded_duration} != expected duration {expected_duration}")
    (output / "video_frames.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in frame_rows) + "\n",
        encoding="utf-8")
    report = {
        "result": "PASS",
        "video": str(video_path),
        "replay": str(bundle.root),
        "trajectory": str(bundle.trajectory_path),
        "encoded_frame_count": encoded_count,
        "source_frame_count": len(bundle.frames),
        "resolution": list(composed.size),
        "source_resolution": [bundle.width, bundle.height],
        "rgb_resolution": [bundle.width, bundle.height],
        "thermal_resolution": [bundle.thermal_width, bundle.thermal_height],
        "video_duration_s": encoded_duration,
        "trajectory_span_s": bundle.frames[-1].simulation_time_s - bundle.frames[0].simulation_time_s,
        "final_frame_hold_s": bundle.frames[-1].duration_s,
        "encoded_pts_s": encoded_pts,
        "encoded_timestamps_verified": True,
        "no_inserted_or_interpolated_frames": True,
        "sample_interval_s": bundle.metadata.get("sample_interval_s"),
        "task_success": (True if bundle.metadata['trajectory_provenance']['task_result']['status'] == 'success'
                         else False if bundle.metadata['trajectory_provenance']['task_result']['status'] == 'failed'
                         else None),
        "task": bundle.metadata['trajectory_provenance']['task'],
        "task_result": bundle.metadata['trajectory_provenance']['task_result'],
        "trajectory_provenance": bundle.metadata['trajectory_provenance'],
        "source_files_unchanged": source_files_unchanged,
        "strict_sync_verified": bundle.strict_sync_verified,
        "strict_per_frame_pixel_sync_verified": bool(bundle.frames) and all(
            frame.strict_sync_verified for frame in bundle.frames),
        "sync_status": bundle.metadata.get(
            "sync_status", "inherited replay status; video generation adds no synchronization evidence"),
        "sync_basis": bundle.metadata.get("sync_basis", "not_verified"),
        "sync_scope_failures": bundle.metadata.get("sync_scope_failures", []),
        "sync_validation_status": bundle.metadata.get("sync_validation_status", "not_verified"),
        "sync_validation_scope": bundle.metadata.get("sync_validation_scope"),
        "sync_validation_evidence": bundle.metadata.get("sync_validation_evidence"),
        "data_usage": "pilot/development data",
        "source_data_usage": bundle.metadata.get("data_usage"),
        "source_sync_status": bundle.metadata.get("sync_status"),
        "frame_metadata": bundle.metadata.get("frame_metadata"),
        "trajectory_sha256": _sha256(bundle.trajectory_path),
        "replay_metadata_sha256": _sha256(bundle.root / "replay_metadata.json"),
        "source_input_hashes": source_hashes_before,
        "video_sha256": _sha256(video_path),
        "video_encoding": "H.264 lossy CRF18/yuv420p; thermal is nearest-neighbor fitted for display only",
        "ffmpeg": str(encoder),
        "ffmpeg_sha256": _sha256(encoder),
        "ffmpeg_command_scope": "portable project-local encoder; no system installation",
    }
    (output / "video_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=None)
    args = parser.parse_args(argv)
    report = generate_video(args.replay, args.out, ffmpeg=args.ffmpeg)
    print(json.dumps({key: report[key] for key in (
        "result", "video", "encoded_frame_count", "resolution", "video_duration_s",
        "source_files_unchanged", "encoded_timestamps_verified")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
