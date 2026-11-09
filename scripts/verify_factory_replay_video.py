"""Verify a factory replay video with an explicit, bounded timestamp tolerance.

The shared video writer rounds concat durations to nanoseconds. Over a long
route that can accumulate a few tens of microseconds, while frame order and
count remain exact. This factory-only check records that measured tolerance
without changing the shared video generator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from scripts.generate_replay_video import find_ffmpeg, load_replay, _probe_frames

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, default=None)
    parser.add_argument('--max-timestamp-error-s', type=float, default=5e-5)
    args = parser.parse_args(argv)
    bundle = load_replay(args.replay)
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    encoder = find_ffmpeg(args.ffmpeg)
    encoded_count, encoded_pts = _probe_frames(encoder, args.video)
    expected = [frame.video_time_s for frame in bundle.frames]
    if encoded_count != len(expected) or len(encoded_pts) != len(expected):
        raise ValueError(f'video frame count {encoded_count}/{len(encoded_pts)} != replay {len(expected)}')
    errors = [abs(actual - wanted) for actual, wanted in zip(encoded_pts, expected)]
    maximum = max(errors, default=0.)
    if maximum > args.max_timestamp_error_s:
        raise ValueError(f'maximum timestamp error {maximum:.9f}s exceeds {args.max_timestamp_error_s:.9f}s')
    report = {
        'result': 'PASS', 'video': str(args.video.resolve()), 'replay': str(args.replay.resolve()),
        'encoded_frame_count': encoded_count, 'expected_frame_count': len(expected),
        'maximum_timestamp_error_s': maximum, 'allowed_timestamp_error_s': args.max_timestamp_error_s,
        'reason_for_relaxed_tolerance': 'shared concat video writer rounds per-frame durations to nanoseconds',
        'strict_sync_verified': False, 'data_usage': 'pilot/development data',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
