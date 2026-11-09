"""Capture one static RGB + thermal frame from the current Crazyflie stage.

Run with ``D:\\isaacsim\\python.bat``.  The stage is opened read-only for this process and is
never saved.  This is the first integration step: the existing RGB camera is also the thermal
camera source so both products share exact pose and intrinsics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import traceback

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
THERMAL_ROOT = PROJECT_ROOT / "third_party" / "isaac-thermal-camera-simulation"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THERMAL_ROOT / "src"))

from scripts.crazyflie_thermal_capture import camera_metadata, validate_frame_contract
from scripts.thermal_config import DEFAULT_THERMAL_CONFIG, load_thermal_config
from scripts.thermal_scene_config import apply_thermal_config, renderable_paths
from scripts.scene_config import DEFAULT_SCENE_CONFIG, load_scene_config
from scripts.thermal_imaging import display_from_outputs, imaging_metadata


DEFAULT_STAGE = PROJECT_ROOT / "lesson_crazyflie.usd"
DEFAULT_SENSOR = THERMAL_ROOT / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
DEFAULT_SCENE = THERMAL_ROOT / "configs" / "scenes" / "quad_flight_clear_noon.yaml"
DEFAULT_CAMERA = "/World/Crazyflie/body/Camera_mount/Camera"


def _hash_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wait_for_stage(app: object, context: object, stage_path: pathlib.Path, camera_path: str) -> object:
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"Could not open stage: {stage_path}")
    for _ in range(60):
        app.update()
        stage = context.get_stage()
        if stage is not None and stage.GetPrimAtPath(camera_path).IsValid():
            return stage
    raise RuntimeError(f"Stage did not expose {camera_path} after loading")


def _build_camera(stage: object, sensor_path: pathlib.Path, scene_path: pathlib.Path, camera_path: str,
                  *, capture_rgb: bool = True, sensor_config=None,
                  thermal_config_path=DEFAULT_THERMAL_CONFIG,
                  flat_field_enabled=False, sensor_seed=0,
                  material_cache_enabled=False, material_directory=None):
    from scripts.city_material_partition import verify_overlay_source
    verify_overlay_source(stage)
    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import load_sensor_config
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
    from irsim.scene import Scene
    from scripts.thermal_runtime import camera_class
    IrCamera = camera_class(material_cache_enabled=material_cache_enabled)

    thermal_config = load_thermal_config(thermal_config_path)
    sensor = load_sensor_config(sensor_path) if sensor_config is None else sensor_config
    spec = sensor.sensor
    lut_dir = THERMAL_ROOT / "data" / "lut"
    lut = load_band_lut_for_config(sensor, lut_dir)
    response = load_band_response_for_config(sensor, THERMAL_ROOT / "data")
    skylight = skylight_for_sensor(sensor, sensor.sensor.quantity)
    scene = Scene.from_file(
        scene_path,
        {spec.band.band_id: lut},
        responses={spec.band.band_id: response},
        quantity=sensor.sensor.quantity,
        skylights={spec.band.band_id: skylight},
    )

    # User-owned YAML drives both captures and preview; USD and vendor files stay untouched.
    paths = renderable_paths(stage)
    from scripts.project_materials import load_project_material_library
    library = load_project_material_library(material_directory)
    materials = MaterialTable.from_library(library, spec.band.band_id)
    scene, prim_to_target, resolutions, config_report = apply_thermal_config(
        scene, thermal_config, paths, materials.names)
    unresolved = len(config_report['fallback_paths'])
    fallback_material = thermal_config.default_material
    print('[thermal-config] loaded ' + thermal_config.source_path, flush=True)
    print('[thermal-config] binding counts: ' + json.dumps(config_report['binding_counts']) +
          f'; defaulted geometry records: {unresolved}', flush=True)
    for name in config_report['unmatched_bindings']:
        binding = next(item for item in thermal_config.bindings if item.name == name)
        print(f'[thermal-config] WARNING: {name} at {binding.prim_path} matched no geometry; '
              'check the full Prim path and include_children.', flush=True)
    pipeline = PipelineConfig.from_sensor(
        sensor,
        materials,
        lut,
        sky=scene.sky_models[spec.band.band_id],
        atmosphere=scene.layered,
        flat_field_enabled=flat_field_enabled,
        sensor_seed=sensor_seed,
    )
    camera = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=prim_to_target,
        resolutions=resolutions,
        camera_path=camera_path,
        capture_rgb=capture_rgb,
        strict_materials=False,
        strict_thermal_nodes=False,
        strict_patch_coverage=False,
        device="cuda:0",
    )
    camera.thermal_config_report = config_report
    config_report['material_library_sha256'] = library.content_hash()
    config_report['material_directory'] = None if material_directory is None else str(material_directory)
    return camera, sensor, unresolved, fallback_material


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_SCENE_CONFIG,
                        help="single YAML file containing stage, drone and sensor paths")
    parser.add_argument("--stage", type=pathlib.Path, default=None)
    parser.add_argument("--sensor", type=pathlib.Path, default=None)
    parser.add_argument("--scene", type=pathlib.Path, default=None)
    parser.add_argument("--thermal-config", type=pathlib.Path, default=None,
                        help="per-object Celsius temperatures and infrared material bindings")
    parser.add_argument("--camera-path", default=None)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--settle", type=int, default=8)
    parser.add_argument("--rt-subframes", type=int, default=1)
    args = parser.parse_args(argv)
    config = load_scene_config(args.config)
    from scripts.thermal_display import load_display_profile
    display_profile = (None if config.sensors.display_config is None else
                       load_display_profile(config.sensors.display_config))
    stage_path = args.stage or config.scene.stage
    sensor = args.sensor or config.sensors.sensor_config
    scene = args.scene or config.scene.thermal_scene
    thermal_config = args.thermal_config or config.sensors.thermal_config
    camera_path = args.camera_path or config.drone.camera_path
    load_thermal_config(thermal_config)  # Fail early, before the costly Kit startup.
    stage_hash_before = _hash_file(stage_path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.out or PROJECT_ROOT / "validation" / f"crazyflie-thermal-static-{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    # Isaac/Omniverse modules must only be imported after Kit is booted.
    from isaacsim import SimulationApp
    from irsim_isaac.env import simulation_app_config
    app = SimulationApp(simulation_app_config())
    camera = None
    status = 1
    try:
        from irsim.io.dataset import FrameWriter
        from irsim.config.loader import band_hash, config_hash
        from irsim_isaac.pipeline.gbuffer_isaac import camera_pose
        import omni.usd

        context = omni.usd.get_context()
        print(f"opening stage: {stage_path}", flush=True)
        stage = _wait_for_stage(app, context, stage_path, camera_path)
        prim = stage.GetPrimAtPath(camera_path)
        if not prim.IsValid():
            raise RuntimeError(f"Camera prim not found: {camera_path}")
        camera, sensor_model, unresolved, fallback_material = _build_camera(
            stage, sensor, scene, camera_path,
            thermal_config_path=thermal_config,
            flat_field_enabled=config.sensors.flat_field_enabled,
            sensor_seed=config.sensors.sensor_seed,
            material_directory=config.sensors.material_directory,
        )
        print(f"building shared RGB/thermal camera: {camera_path}", flush=True)
        camera.open(settle_frames=args.settle, rt_subframes=args.rt_subframes)
        camera.refresh_pose()
        print("capturing one frame", flush=True)
        outputs = display_from_outputs(camera.config,
                                       camera.get_outputs(step=True, rt_subframes=args.rt_subframes),
                                       display_profile)
        frame = camera.last_frame
        rgb = None if frame is None else frame.rgb
        if rgb is None:
            raise RuntimeError(f"IrCamera did not produce companion RGB: {camera.rgb_problem}")
        expected_shape = (sensor_model.sensor.fpa.height, sensor_model.sensor.fpa.width)
        contract = validate_frame_contract(
            rgb=rgb,
            radiance=outputs.radiance,
            apparent_t=outputs.apparent_t,
            dn16=outputs.dn16,
            display8=outputs.display8,
            expected_shape=expected_shape,
        )
        position, camera_to_world = camera_pose(camera_path, stage=stage)
        stage_hash_after = _hash_file(stage_path)
        if stage_hash_after != stage_hash_before:
            raise RuntimeError('Source stage changed during capture; check external edits')
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
                "thermal_config_sha256": camera.thermal_config_report['sha256'],
                "thermal_imaging": imaging_metadata(camera.config, camera.state, display_profile),
                "thermal_target_temperatures_k": {
                    name: float(value) for name, value in camera.bridge.temperatures().items()
                    if name in camera.thermal_config_report['initial_target_temperatures_k']
                },
                **camera_metadata(
                    camera_path=camera_path,
                    stage_path=str(stage_path),
                    sensor_config=str(sensor),
                    camera_position=position,
                    capture_time_s=camera.last_frame_t_s,
                ),
            },
        )
        record = writer.write(
            outputs,
            frame_index=0,
            t_s=camera.last_frame_t_s,
            extra_planes={"rgb": rgb},
            extra_metadata={"camera_to_world": np.asarray(camera_to_world).tolist()},
        )
        summary = {
            "result": "PASS",
            "record": None if record is None else record.metadata,
            "output_directory": str(output),
            "warning": "This is a software thermal prototype; apparent_t is not calibrated surface temperature.",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        status = 0
    except Exception as exc:  # noqa: BLE001 - capture must leave an actionable report
        diagnostics = {}
        if camera is not None:
            last = getattr(camera, "_last", None)
            if last is not None:
                diagnostics["unmapped_gbuffer_pixels"] = int(np.asarray(last.unmapped).sum())
                diagnostics["gbuffer_pixels"] = int(np.asarray(last.unmapped).size)
                diagnostics["label_count"] = len(getattr(last, "labels", {}))
                diagnostics["labels_sample"] = {
                    str(key): value for key, value in list(getattr(last, "labels", {}).items())[:30]
                }
            if getattr(camera, "last_frame", None) is not None:
                thermal = getattr(camera, "last_frame", None)
                diagnostics["rgb_shape"] = None if thermal.rgb is None else list(np.asarray(thermal.rgb).shape)
            diagnostics["resolution_count"] = len(getattr(camera, "resolutions", []))
        failure = {
            "result": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "output_directory": str(output),
            "diagnostics": diagnostics,
        }
        (output / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        traceback.print_exc()
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr, flush=True)
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:  # noqa: BLE001 - preserve the original capture status
                traceback.print_exc()
        app.close(exit_code=status)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
