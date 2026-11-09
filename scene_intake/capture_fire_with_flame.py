"""森林火场景静态取图（带火焰热代面）。

为什么需要这个入口：作者的红外预览会先把透明火舌换成"不透明热代面"，否则 RTX 几何
AOV 里根本没有火焰。本项目既有的 `capture_crazyflie_thermal_static.py` 没有这一步，
所以直接用它拍出来的红外只有桶，没有火。

本脚本复用既有脚本的构件（`_build_camera` / `FrameWriter` / 展示管线），只在渲染前后
套上作者包里的 `fire_thermal_proxy.thermal_flame_surfaces`。**不改动任何既有脚本**，
也不改 USD（代面只存在于会话层）。

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\capture_fire_with_flame.py --out D:\\IsaacSim_Demo\\validation\\forest-fire-flame-01
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT = pathlib.Path(r"D:\IsaacSim_Demo")
SCENE_TOOLS = PROJECT / "forest_fire_scene_local" / "scripts"
for entry in (PROJECT, PROJECT / "third_party" / "isaac-thermal-camera-simulation" / "src", SCENE_TOOLS):
    sys.path.insert(0, str(entry))

from isaacsim import SimulationApp  # noqa: E402
from irsim_isaac.env import simulation_app_config  # noqa: E402

app = SimulationApp(simulation_app_config())

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from irsim.config.loader import band_hash, config_hash  # noqa: E402
from irsim.io.dataset import FrameWriter  # noqa: E402
from irsim_isaac.pipeline.gbuffer_isaac import camera_pose  # noqa: E402
from scripts.capture_crazyflie_thermal_static import _build_camera, _hash_file, _wait_for_stage  # noqa: E402
from scripts.crazyflie_thermal_capture import camera_metadata, validate_frame_contract  # noqa: E402
from scripts.scene_config import load_scene_config  # noqa: E402
from scripts.thermal_display import load_display_profile  # noqa: E402
from scripts.thermal_imaging import display_from_outputs, imaging_metadata  # noqa: E402
try:  # 优先用本项目自己的副本，脱离作者脚本目录
    from scene_intake.fire_thermal_proxy import thermal_flame_surfaces  # noqa: E402
except ImportError:  # noqa: E402
    from fire_thermal_proxy import thermal_flame_surfaces  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(PROJECT / "configs" / "forest_fire_scene.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-code", type=float, default=0.0, help="USD 动画帧（24 fps），默认 0")
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--rt-subframes", type=int, default=1)
    args = ap.parse_args()

    config = load_scene_config(args.config)
    display_profile = (None if config.sensors.display_config is None
                       else load_display_profile(config.sensors.display_config))
    stage_path = config.scene.stage
    output = pathlib.Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    stage_hash_before = _hash_file(stage_path)

    context = omni.usd.get_context()
    stage = _wait_for_stage(app, context, stage_path, config.drone.camera_path)
    camera, sensor_model, unresolved, fallback_material = _build_camera(
        stage,
        config.sensors.sensor_config,
        config.scene.thermal_scene,
        config.drone.camera_path,
        thermal_config_path=config.sensors.thermal_config,
        flat_field_enabled=config.sensors.flat_field_enabled,
        sensor_seed=config.sensors.sensor_seed,
        material_directory=config.sensors.material_directory,
    )

    # 固定动画时刻，保证 RGB 与红外、以及多次运行之间可比（24 fps）
    import omni.timeline
    omni.timeline.get_timeline_interface().set_current_time(args.time_code / 24.0)

    # 场景里记录的推荐渲染参数（customLayerData.renderPreset）。Isaac 本身不读它，
    # 但我们的入口按它设置采样数，让"参数随文件走"这件事在本工具链里成立。
    preset = (stage.GetRootLayer().customLayerData or {}).get("renderPreset")
    if preset:
        spp = preset.get("samplesPerPixel")
        subframes = preset.get("rtSubframes")
        if spp and args.settle:
            pass  # settle 是相机稳定帧数，不覆盖
        if subframes and args.rt_subframes == 1:
            args.rt_subframes = int(subframes)
        print(f"[renderPreset] {preset.get('renderer')} spp={spp} subframes={args.rt_subframes}", flush=True)

    camera.open(settle_frames=args.settle, rt_subframes=args.rt_subframes)
    camera.refresh_pose()

    # 第一遍：不套代面，取 RGB（透明火舌正常显示）
    camera.get_outputs(step=True, rt_subframes=args.rt_subframes)
    frame_rgb = camera.last_frame
    rgb = None if frame_rgb is None else frame_rgb.rgb
    if rgb is None:
        raise RuntimeError(f"IrCamera did not produce companion RGB: {camera.rgb_problem}")

    # 第二遍：套上不透明热代面，取红外（RGB 这一遍不用）
    with thermal_flame_surfaces(stage, args.time_code):
        outputs = display_from_outputs(
            camera.config,
            camera.get_outputs(step=True, rt_subframes=args.rt_subframes),
            display_profile,
        )
        expected = (sensor_model.sensor.fpa.height, sensor_model.sensor.fpa.width)
        contract = validate_frame_contract(rgb=rgb, radiance=outputs.radiance,
                                          apparent_t=outputs.apparent_t, dn16=outputs.dn16,
                                          display8=outputs.display8, expected_shape=expected)
        position, camera_to_world = camera_pose(config.drone.camera_path, stage=stage)

    stage_hash_after = _hash_file(stage_path)
    if stage_hash_after != stage_hash_before:
        raise RuntimeError("Source stage changed during capture; check external edits")

    writer = FrameWriter(
        directory=output,
        config_hash=config_hash(sensor_model),
        band_hash=band_hash(sensor_model),
        quantity=camera.config.quantity,
        start_utc=camera.scene.weather.start_utc,
        metadata={
            "scene_config_path": str(config.source_path),
            "scene_config_sha256": config.sha256,
            "drone_prim_path": config.drone.prim_path,
            "drone_body_path": config.drone.body_path,
            "stage_sha256_before": stage_hash_before,
            "stage_sha256_after": stage_hash_after,
            "camera_contract": contract,
            "unresolved_material_records": unresolved,
            "fallback_material": fallback_material,
            "fallback_material_scope": "transient_capture_process_only",
            "thermal_configuration": camera.thermal_config_report,
            "thermal_config_sha256": camera.thermal_config_report["sha256"],
            "thermal_imaging": imaging_metadata(camera.config, camera.state, display_profile),
            "thermal_target_temperatures_k": {
                name: float(value) for name, value in camera.bridge.temperatures().items()
                if name in camera.thermal_config_report["initial_target_temperatures_k"]
            },
            "flame_thermal_proxy": {"time_code": args.time_code,
                                    "source": "forest_fire_scene_local/scripts/fire_thermal_proxy.py",
                                    "note": "opaque 650 C silhouettes in the session layer; not written to USD"},
            **camera_metadata(camera_path=config.drone.camera_path, stage_path=str(stage_path),
                              sensor_config=str(config.sensors.sensor_config),
                              camera_position=position, capture_time_s=camera.last_frame_t_s),
        },
    )
    record = writer.write(outputs, frame_index=0, t_s=camera.last_frame_t_s,
                          extra_planes={"rgb": rgb},
                          extra_metadata={"camera_to_world": np.asarray(camera_to_world).tolist()})

    flame_pixels = int(np.count_nonzero(outputs.apparent_t > 800.0))
    summary = {
        "result": "PASS",
        "record": None if record is None else record.metadata,
        "output_directory": str(output),
        "flame_pixels_above_800K": flame_pixels,
        "apparent_t_max_k": float(np.max(outputs.apparent_t)),
        "warning": "Software thermal prototype; apparent_t is not calibrated surface temperature.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"result": "PASS", "flame_pixels_above_800K": flame_pixels,
                      "apparent_t_max_k": float(np.max(outputs.apparent_t)),
                      "output": str(output)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        app.close()
