"""既有采集/回放流程的"零改动"护栏。

给正在使用的文件拍哈希基线，之后任何时候都能验证"我们没动过它们"。
只读；不写任何被监视的目录。

用法：
    python guard_pipeline.py baseline     # 首次建立基线
    python guard_pipeline.py verify       # 与基线比对（有差异时退出码 2）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
STORE = PROJECT / "scene_intake" / "guard" / "baseline.json"

# 采集/回放真正依赖的东西。新增目录不在此列，所以我们的加入不会被误判为改动。
WATCHED = [
    "run_collect_live.bat",
    "run_collect_replay.bat",
    "README.md",
    "scripts",
    "configs",
    "docs",
    "third_party/isaac-thermal-camera-simulation",
]
SKIP_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
SKIP_SUFFIX = {".pyc", ".pyo"}


def iter_files(rel: str):
    base = PROJECT / rel
    if base.is_file():
        yield base
        return
    if not base.is_dir():
        return
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(PROJECT).parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIX:
            continue
        yield path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot() -> dict:
    files = {}
    for rel in WATCHED:
        for path in iter_files(rel):
            relpath = path.relative_to(PROJECT).as_posix()
            st = path.stat()
            files[relpath] = {"sha256": sha256(path), "bytes": st.st_size}
    return {
        "created_utc": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project": str(PROJECT),
        "watched": WATCHED,
        "count": len(files),
        "files": files,
    }


def cmd_baseline(_args) -> int:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    data = snapshot()
    STORE.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    print(f"baseline written: {STORE}")
    print(f"watched roots: {', '.join(WATCHED)}")
    print(f"files: {data['count']}")
    return 0


def cmd_rebaseline(args) -> int:
    """把当前状态作为新基线，旧基线归档。用于"外部确实改过、且已确认"之后。"""
    if STORE.exists():
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        archive = STORE.with_name(f"baseline-{stamp}.json")
        STORE.replace(archive)
        print(f"previous baseline archived: {archive}")
    data = snapshot()
    data["note"] = args.note or ""
    STORE.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    print(f"baseline replaced: {STORE}")
    print(f"note: {data['note'] or '(none)'}")
    print(f"files: {data['count']}")
    return 0


def cmd_verify(_args) -> int:
    if not STORE.exists():
        print("no baseline; run `python guard_pipeline.py baseline` first", file=sys.stderr)
        return 3
    base = json.loads(STORE.read_text(encoding="utf-8"))
    now = snapshot()
    old_files, new_files = base["files"], now["files"]
    added = sorted(set(new_files) - set(old_files))
    removed = sorted(set(old_files) - set(new_files))
    changed = sorted(p for p in set(old_files) & set(new_files) if old_files[p]["sha256"] != new_files[p]["sha256"])

    print(f"baseline: {base['created_utc']}  ({base['count']} files)")
    print(f"now     : {now['created_utc']}  ({now['count']} files)")
    print(f"added={len(added)} removed={len(removed)} changed={len(changed)}")
    for label, items in (("ADDED", added), ("REMOVED", removed), ("CHANGED", changed)):
        for p in items[:20]:
            print(f"  {label}: {p}")
        if len(items) > 20:
            print(f"  ... and {len(items) - 20} more {label.lower()}")

    # 新增文件不构成破坏（既有脚本按路径读取，不会因为多出文件而改变行为）；
    # 真正危险的是"改动"和"删除"。
    if not (removed or changed):
        print(f"RESULT: SAFE (added {len(added)} new file(s), 0 modified, 0 removed)")
        return 0
    print("RESULT: DRIFT DETECTED")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["baseline", "verify", "rebaseline"])
    ap.add_argument("--note", default="", help="rebaseline 时记录原因")
    args = ap.parse_args()
    if args.action == "baseline":
        return cmd_baseline(args)
    if args.action == "rebaseline":
        return cmd_rebaseline(args)
    return cmd_verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
