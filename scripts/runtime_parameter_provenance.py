"""Opt-in recording/replay identity for editable project thermal dependencies."""
import hashlib
from pathlib import Path
from scripts.project_materials import load_project_material_library,VENDOR


def thermal_input_snapshot(config):
    library=load_project_material_library(config.sensors.material_directory)
    from irsim.config.loader import resolve_data_dir
    from irsim.config.scene import load_scene_config
    from irsim.thermal.weather_io import load_weather_csv
    environment=load_scene_config(config.scene.thermal_scene)
    weather_path=(resolve_data_dir()/environment.scene.weather_file).resolve()
    weather=load_weather_csv(weather_path)
    files={config.sensors.sensor_config,config.sensors.thermal_config,config.scene.thermal_scene,weather_path}
    if getattr(config.sensors,'display_config',None) is not None:files.add(config.sensors.display_config)
    files.update((VENDOR/'configs/materials').glob('*.yaml'))
    for material in library.values():
        files.add(material.path)
        if material.spectrum is not None:files.add(material.spectrum.path)
        if material.n_k_path is not None:files.add(material.n_k_path)
    return {'schema_version':1,'material_library_sha256':library.content_hash(),'weather_content_sha256':weather.content_hash,
        'file_sha256':{str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(files)}}


def validate_thermal_inputs(config,recorded):
    if recorded is None and config.sensors.material_directory is None:return None
    if not isinstance(recorded,dict):raise ValueError('Missing recorded project thermal input snapshot; original inputs or a new recording are required')
    actual=thermal_input_snapshot(config)
    if actual!=recorded:raise ValueError('Recorded thermal/material/weather inputs differ; restore original inputs or make a new recording')
    return actual
