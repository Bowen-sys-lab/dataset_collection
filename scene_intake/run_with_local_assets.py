"""以"离线可用的资源根"跑既有脚本，绕开启动时的 S3 资源检查。

本机没有外网，Isaac 的 `isaacsim.robot.policy.examples` 扩展在启动时会去查
omniverse S3 资源根，偶发抛 `Could not find assets root folder` 让进程直接失败。
给 SimulationApp 注入 `--/persistent/isaac/asset_root/default=<本地目录>` 即可绕过；
本目录下建一个只含 `Isaac/` 的空目录就够，扩展只做存在性检查。

不修改被调脚本：在它 `from isaacsim import SimulationApp` 之前替换该类即可。

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\run_with_local_assets.py scripts\\replay_trajectory_highres.py -- <原脚本参数...>
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
ASSETS = PROJECT / "validation" / "isaac_local_assets"
(ASSETS / "Isaac").mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 3 or sys.argv[2] != "--":
    raise SystemExit(__doc__)

target = Path(sys.argv[1])
if not target.is_absolute():
    target = PROJECT / target
script_args = sys.argv[3:]

sys.path.insert(0, str(PROJECT))
import isaacsim  # noqa: E402

_Original = isaacsim.SimulationApp


class _LocalAssetsApp(_Original):  # type: ignore[misc,valid-type]
    def __init__(self, config=None, *args, **kwargs):
        cfg = dict(config or {})
        extra = list(cfg.get("extra_args", []))
        extra.append("--/persistent/isaac/asset_root/default=" + str(ASSETS).replace("\\", "/"))
        cfg["extra_args"] = extra
        super().__init__(cfg, *args, **kwargs)


isaacsim.SimulationApp = _LocalAssetsApp
sys.argv = [str(target)] + script_args
runpy.run_path(str(target), run_name="__main__")
