"""Bind user YAML to irsim in memory; never author USD or edit vendor files."""
from dataclasses import asdict, replace

from scripts.thermal_config import ThermalConfigError


def renderable_paths(stage, root='/World'):
    """Include USD instance proxies: renderer IDs name these actual surfaces.

    This local adapter only reads geometry. It does not de-instance assets or
    guess that an entire multi-material instance has one infrared material.
    """
    from pxr import Usd, UsdGeom
    start = stage.GetPrimAtPath(root)
    if not start or not start.IsActive():
        raise ThermalConfigError(f'No active geometry root at {root}')
    return [str(prim.GetPath()) for prim in Usd.PrimRange(start, Usd.TraverseInstanceProxies())
            if prim.IsA(UsdGeom.Gprim) and
            UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible]


def apply_thermal_config(scene, config, prim_paths, material_names):
    from irsim.materials.mapping import Resolution
    from irsim.thermal.aerial import airframe_solver
    from irsim.thermal.solvers import PrescribedSolver

    material_ids = {name: index for index, name in enumerate(material_names) if index > 0}
    requested_materials = {config.default_material, *(binding.material for binding in config.bindings)}
    unknown = requested_materials - material_ids.keys()
    if unknown:
        raise ThermalConfigError(
            f'{config.source_path}: unknown infrared materials {sorted(unknown)}. '
            f'Available materials: {sorted(material_ids)}')
    collisions = {item.name for item in config.targets} & (set(scene.targets) | set(scene.thermal_surfaces))
    if collisions:
        raise ThermalConfigError(f'{config.source_path}: target names already used by base scene: '
                                 f'{sorted(collisions)}; choose distinct names in your YAML')

    targets = dict(scene.targets)
    for item in config.targets:
        if item.mode == 'fixed':
            solver = PrescribedSolver(
                [float(scene.weather.time_s[0]), float(scene.weather.time_s[-1])],
                [item.temperature_k, item.temperature_k], t0_s=scene.t0_s)
        else:
            solver = airframe_solver(scene.weather, offset_k=item.offset_k)
            # The weather file starts before the scene; prime at scene time.
            solver.advance(scene.t0_s, 0.0)
        targets[item.name] = solver
    configured_scene = replace(scene, targets=targets)

    mapping, resolutions, assignments, fallback_paths = {}, [], [], []
    counts = {binding.name: 0 for binding in config.bindings}
    for path in prim_paths:
        binding = config.binding_for(path)
        target = config.default_target if binding is None else binding.target
        material = config.default_material if binding is None else binding.material
        if binding is None:
            fallback_paths.append(path)
        else:
            counts[binding.name] += 1
        mapping[path] = target
        resolutions.append(Resolution(path, material, material_ids[material], 'override',
                                      'thermal_yaml:' + (binding.name if binding else 'defaults')))
        assignments.append({'prim_path': path, 'target': target, 'material': material,
                            'binding': None if binding is None else binding.name})
    if not mapping:
        raise ThermalConfigError('No renderable surfaces found below /World')
    report = {
        'path': config.source_path, 'sha256': config.sha256, 'schema_version': config.schema_version,
        'input_temperature_unit': 'C', 'solver_temperature_unit': 'K',
        'defaults': {'target': config.default_target, 'material': config.default_material},
        'targets': [asdict(item) for item in config.targets],
        'bindings': [asdict(item) for item in config.bindings],
        'binding_counts': counts, 'fallback_paths': fallback_paths,
        'unmatched_bindings': [name for name, count in counts.items() if count == 0],
        'assignments': assignments,
        'initial_target_temperatures_k': {item.name: float(targets[item.name].temperature())
                                          for item in config.targets},
        'temperature_model': 'prescribed per-object temperature; no energy balance or heat conduction',
    }
    return configured_scene, mapping, resolutions, report
