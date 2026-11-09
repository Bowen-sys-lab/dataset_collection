"""Capture fixed RGB views, probe the RGB save path, and map RTX UV warnings.

All scene edits are authored into temporary anonymous session layers. The root
USD, thermal source tree, preview renderer settings, and flight controller are
not changed by this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/isaac-thermal-camera-simulation/src'))

from scripts.rgb_scene_diagnostics import (  # noqa: E402
    color_patch_channels,
    compare_rgb_buffers,
    local_transform_from_world,
    parse_uv_warnings,
    project_world_bbox_to_image,
    project_patch_roi,
    region_statistics,
    uv_signature_matches,
    uv_warning_sources_inspected,
    world_prim_range,
)
from scripts.scene_config import DEFAULT_SCENE_CONFIG, load_scene_config  # noqa: E402
from scripts.trajectory_recorder import read_trajectory  # noqa: E402


COLOR_SWATCHES = (
    ('red', (1.0, 0.0, 0.0)),
    ('green', (0.0, 1.0, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('yellow', (1.0, 1.0, 0.0)),
    ('cyan', (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)),
    ('neutral_gray', (0.18, 0.18, 0.18)),
)
CHART_DISTANCE = 2.0
CHART_PATCH_SIZE = (0.24, 0.24)
CHART_X_CENTERS = (-0.72, -0.48, -0.24, 0.0, 0.24, 0.48, 0.72)
KIT_LOG_ROOT = Path(r'D:\isaacsim\kit\logs\Kit\Isaac-Sim Python\6.1')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frames', default='0,80,160,262',
                        help='comma-separated trajectory indices; defaults to four spaced views')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--kit-log', type=Path, default=None,
                        help='optional Kit log to parse; otherwise use the newest log from this run')
    return parser


def parse_frame_indices(value, count):
    try:
        indices = [int(part.strip()) for part in value.split(',') if part.strip()]
    except ValueError as exc:
        raise ValueError('--frames must be comma-separated integers') from exc
    if not indices:
        raise ValueError('--frames must contain at least one index')
    if len(set(indices)) != len(indices):
        raise ValueError('--frames contains duplicate indices')
    if any(index < 0 or index >= count for index in indices):
        raise ValueError(f'--frames indices must be in [0, {count - 1}]')
    return indices


def _simple_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, 'path') and hasattr(value, 'resolvedPath'):
        return {'authored_path': str(value.path), 'resolved_path': str(value.resolvedPath)}
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return str(value)


def _local_asset_exists(resolved_path):
    text = str(resolved_path or '')
    if not text or '://' in text:
        return None
    path = Path(text)
    return path.is_file() if path.is_absolute() else None


def _instance_source_records(prim):
    records = []
    for spec in prim.GetPrimStack():
        layer = spec.layer
        records.append({
            'composed_spec_path': str(spec.path),
            'layer_identifier': str(layer.identifier),
            'layer_real_path': str(getattr(layer, 'realPath', '') or ''),
        })
    references = []
    try:
        reference_op = prim.GetMetadata('references')
        for reference in reference_op.GetAppliedItems():
            references.append({
                'asset_path': str(reference.assetPath),
                'prim_path': str(reference.primPath),
            })
    except Exception:
        pass
    return {'prim_stack': records, 'authored_references': references}


def _bound_material_report(prim):
    from pxr import Sdf, Usd, UsdShade

    try:
        material, binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    except Exception as exc:
        return {'error': str(exc), 'material_path': None, 'shaders': []}
    if not material or not material.GetPrim().IsValid():
        return {'material_path': None, 'binding_relationship': None, 'shaders': []}

    shader_records = []
    for shader_prim in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(shader_prim)
        if not shader:
            continue
        shader_record = {
            'prim_path': str(shader_prim.GetPath()),
            'shader_id': str(shader.GetIdAttr().Get() or ''),
            'source_assets': [],
            'color_and_uv_inputs': {},
        }
        try:
            for source_type in ('mdl', 'glslfx'):
                source_asset = shader.GetSourceAsset(source_type)
                if source_asset and source_asset.path:
                    shader_record['source_assets'].append({
                        'source_type': source_type,
                        'authored_path': str(source_asset.path),
                        'resolved_path': str(source_asset.resolvedPath),
                        'local_file_exists': _local_asset_exists(source_asset.resolvedPath),
                    })
        except Exception:
            pass
        for shader_input in shader.GetInputs():
            base_name = str(shader_input.GetBaseName())
            value = shader_input.Get()
            is_asset = isinstance(value, Sdf.AssetPath)
            lower_name = base_name.lower()
            interesting = is_asset or any(token in lower_name for token in (
                'diffusecolor', 'basecolor', 'emissivecolor', 'varname', 'texture', 'file', 'color'
            ))
            if not interesting:
                continue
            converted = _simple_value(value)
            if is_asset:
                converted['local_file_exists'] = _local_asset_exists(value.resolvedPath)
            shader_record['color_and_uv_inputs'][base_name] = converted
        shader_records.append(shader_record)
    return {
        'material_path': str(material.GetPath()),
        'material_name': str(material.GetPrim().GetName()),
        'binding_relationship': str(binding.GetPath()) if binding else None,
        'shaders': shader_records,
    }


def _mesh_report(stage, warning_records, sampled_frames, rgb_camera, width, height,
                 inventory_out=None):
    from pxr import Sdf, Usd, UsdGeom

    grouped = {}
    for warning in warning_records:
        group = grouped.setdefault(warning['mesh_prim_path'], {
            'mesh_prim_path': warning['mesh_prim_path'],
            'prototype_path': warning['prototype_path'],
            'warnings': [],
        })
        group['warnings'].append({
            'primvar_name': warning['primvar_name'],
            'reported_buffer_size': warning['reported_buffer_size'],
            'expected_face_varying_size': warning['expected_face_varying_size'],
        })

    prototype_prims = {}
    try:
        for prototype in stage.GetPrototypes():
            prototype_prims[str(prototype.GetPath())] = prototype
    except Exception:
        pass

    instances_by_prototype = {}
    candidates = world_prim_range(stage, Usd)
    for candidate in candidates:
        try:
            if not candidate.IsInstance():
                continue
            prototype = candidate.GetPrototype()
            if not prototype or not prototype.IsValid():
                continue
            prototype_path = str(prototype.GetPath())
            instances_by_prototype.setdefault(prototype_path, []).append({
                'instance_prim_path': str(candidate.GetPath()),
                **_instance_source_records(candidate),
            })
        except Exception:
            continue

    # RTX uses renderer-owned prototype identifiers that do not survive as USD
    # prim paths. Build a candidate index of composed mesh instances using a
    # geometry/UV signature rather than pretending those internal names exist
    # in the authored USD hierarchy.
    signature_candidates = {}
    mesh_inventory = []
    candidates = world_prim_range(stage, Usd)
    for candidate in candidates:
        if not candidate.IsValid() or not candidate.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(candidate)
        face_counts = mesh.GetFaceVertexCountsAttr().Get() or []
        corner_count = int(sum(int(value) for value in face_counts))
        inventory_record = {
            'prim_path': str(candidate.GetPath()),
            'mesh_name': str(candidate.GetName()),
            'is_instance_proxy': bool(candidate.IsInstanceProxy()),
            'face_count': len(face_counts),
            'corner_count': corner_count,
            'point_count': len(mesh.GetPointsAttr().Get() or []),
            'uv_primvars': {},
        }
        mesh_inventory.append(inventory_record)
        if corner_count == 0:
            continue
        for primvar in UsdGeom.PrimvarsAPI(candidate).GetPrimvars():
            name = str(primvar.GetPrimvarName())
            if not (name == 'st' or name.startswith('st_')):
                continue
            values = primvar.Get()
            if values is None:
                continue
            value_count = len(values)
            indices = primvar.GetIndices() if primvar.IsIndexed() else None
            index_count = None if indices is None else len(indices)
            inventory_record['uv_primvars'][name] = {
                'values_count': value_count,
                'indices_count': index_count,
                'interpolation': str(primvar.GetInterpolation()),
            }
            by_path = signature_candidates.setdefault(corner_count, {})
            mesh_record = by_path.setdefault(str(candidate.GetPath()), {
                'prim': candidate,
                'prim_path': str(candidate.GetPath()),
                'uv_primvars': {},
            })
            mesh_record['uv_primvars'][name] = {
                'values_count': int(value_count),
                'indices_count': None if index_count is None else int(index_count),
                'interpolation': str(primvar.GetInterpolation()),
            }

    if inventory_out is not None:
        Path(inventory_out).write_text(json.dumps(mesh_inventory, indent=2), encoding='utf-8')

    results = []
    for mesh_path, record in grouped.items():
        prototype = prototype_prims.get(record['prototype_path'])
        prim = None
        if prototype:
            relative = mesh_path[len(record['prototype_path']):].strip('/')
            prim = prototype
            for part in relative.split('/') if relative else ():
                prim = prim.GetChild(part)
                if not prim or not prim.IsValid():
                    prim = None
                    break
        record['prototype_exists_in_loaded_stage'] = bool(prototype and prototype.IsValid())
        record['mesh_prim_valid_in_loaded_stage'] = bool(prim and prim.IsValid())
        record['referencing_instances'] = instances_by_prototype.get(record['prototype_path'], [])
        for instance in record['referencing_instances']:
            relative = mesh_path[len(record['prototype_path']):].strip('/')
            instance_mesh_path = str(instance['instance_prim_path']) + ('/' + relative if relative else '')
            proxy = stage.GetPrimAtPath(Sdf.Path(instance_mesh_path))
            instance['source_mesh_prim_path'] = instance_mesh_path
            instance['source_mesh_prim_valid'] = bool(proxy and proxy.IsValid())
            if proxy and proxy.IsValid() and proxy.IsA(UsdGeom.Imageable):
                try:
                    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                               [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                    aligned = bounds.ComputeWorldBound(proxy).ComputeAlignedRange()
                    instance['world_bounds_min'] = [float(value) for value in aligned.GetMin()]
                    instance['world_bounds_max'] = [float(value) for value in aligned.GetMax()]
                    instance['visibility'] = str(UsdGeom.Imageable(proxy).ComputeVisibility(Usd.TimeCode.Default()))
                    instance['view_observations'] = []
                    for frame in sampled_frames:
                        camera_world = frame['rgb_camera_world_matrix']
                        projection = project_world_bbox_to_image(
                            bbox_min=instance['world_bounds_min'],
                            bbox_max=instance['world_bounds_max'],
                            camera_world=camera_world,
                            focal_length_mm=float(rgb_camera.GetFocalLengthAttr().Get()),
                            horizontal_aperture_mm=float(rgb_camera.GetHorizontalApertureAttr().Get()),
                            vertical_aperture_mm=float(rgb_camera.GetVerticalApertureAttr().Get()),
                            width=width, height=height,
                        )
                        instance['view_observations'].append({
                            'trajectory_index': frame['trajectory_index'],
                            **projection,
                            'occlusion_checked': False,
                        })
                except Exception as exc:
                    instance['bounds_error'] = str(exc)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            record['mesh_geometry'] = None
            record['material'] = None
            candidates_by_path = {}
            for warning in record['warnings']:
                warning['signature_candidates'] = []
                for candidate in signature_candidates.get(warning['expected_face_varying_size'], {}).values():
                    for candidate_name, primvar in candidate['uv_primvars'].items():
                        if candidate_name != warning['primvar_name']:
                            continue
                        if not uv_signature_matches(
                                expected_corners=warning['expected_face_varying_size'],
                                values_count=primvar['values_count'],
                                indices_count=primvar['indices_count'],
                                warning_expected=warning['expected_face_varying_size'],
                                warning_buffer=warning['reported_buffer_size']):
                            continue
                        candidate_match = candidates_by_path.setdefault(candidate['prim_path'], {
                            'source_mesh_prim_path': candidate['prim_path'],
                            'source_mesh_instance_proxy': bool(candidate['prim'].IsInstanceProxy()),
                            'face_vertex_corner_count': warning['expected_face_varying_size'],
                            'mesh_name_matches_warning': candidate['prim'].GetName() == mesh_path.rsplit('/', 1)[-1],
                            'matched_warning_primvars': [],
                            'match_count': 0,
                            '_prim': candidate['prim'],
                        })
                        candidate_match['matched_warning_primvars'].append({
                            'warning_primvar_name': warning['primvar_name'],
                            'warning_buffer_size': warning['reported_buffer_size'],
                            'matched_source_primvar_name': candidate_name,
                            'matched_source_values_count': primvar['values_count'],
                            'matched_source_indices_count': primvar['indices_count'],
                            'source_primvar_interpolation': primvar['interpolation'],
                        })
                        candidate_match['match_count'] += 1
                        warning['signature_candidates'].append({
                            'source_mesh_prim_path': candidate['prim_path'],
                            'matched_source_primvar_name': candidate_name,
                            'matched_values_count': primvar['values_count'],
                            'matched_indices_count': primvar['indices_count'],
                        })
            candidate_records = []
            for candidate in candidates_by_path.values():
                source_prim = candidate.pop('_prim')
                source_root = source_prim
                instance_root_path = None
                while source_root and source_root.IsValid():
                    if source_root.IsInstance():
                        instance_root_path = str(source_root.GetPath())
                        break
                    source_root = source_root.GetParent()
                bounds_values, visibility = None, None
                try:
                    visibility = str(UsdGeom.Imageable(source_prim).ComputeVisibility(Usd.TimeCode.Default()))
                    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                                   [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                    aligned = bbox_cache.ComputeWorldBound(source_prim).ComputeAlignedRange()
                    bounds_values = {
                        'min': [float(value) for value in aligned.GetMin()],
                        'max': [float(value) for value in aligned.GetMax()],
                    }
                except Exception:
                    pass
                candidate['source_mesh_instance_root'] = instance_root_path
                candidate['source_prim_visibility'] = visibility
                candidate['source_mesh_world_bounds'] = bounds_values
                candidate['source_material'] = _bound_material_report(source_prim)
                candidate['source_asset'] = _instance_source_records(source_prim)
                candidate['material_subsets'] = [
                    {
                        'prim_path': str(child.GetPath()),
                        'face_count': len(UsdGeom.Subset(child).GetIndicesAttr().Get() or []),
                        'material': _bound_material_report(child),
                    }
                    for child in source_prim.GetChildren() if child.IsA(UsdGeom.Subset)
                ]
                matched_names = {item['warning_primvar_name'] for item in candidate['matched_warning_primvars']}
                candidate['signature_match_strength'] = 'mesh_name_corner_count_and_uv_element_counts'
                candidate['all_warning_uv_primvars_matched'] = matched_names == {
                    item['primvar_name'] for item in record['warnings']}
                candidate['view_observations'] = [
                    {'trajectory_index': frame['trajectory_index'], **project_world_bbox_to_image(
                        bbox_min=bounds_values['min'], bbox_max=bounds_values['max'],
                        camera_world=frame['rgb_camera_world_matrix'],
                        focal_length_mm=float(rgb_camera.GetFocalLengthAttr().Get()),
                        horizontal_aperture_mm=float(rgb_camera.GetHorizontalApertureAttr().Get()),
                        vertical_aperture_mm=float(rgb_camera.GetVerticalApertureAttr().Get()),
                        width=width, height=height,
                    ), 'occlusion_checked': False}
                    for frame in sampled_frames
                ] if bounds_values else []
                candidate_records.append(candidate)
            candidate_records.sort(key=lambda row: (not row['mesh_name_matches_warning'],
                                                    -row['match_count'], row['source_mesh_prim_path']))
            record['source_mapping_status'] = 'signature_candidates_not_renderer_identity'
            record['signature_candidate_count'] = len(candidate_records)
            record['signature_candidates'] = candidate_records[:100]
            results.append(record)
            continue

        mesh = UsdGeom.Mesh(prim)
        face_counts = mesh.GetFaceVertexCountsAttr().Get() or []
        expected_corners = int(sum(int(value) for value in face_counts))
        primvars = {}
        for warning in record['warnings']:
            name = warning['primvar_name']
            primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar(name)
            if not primvar or not primvar.IsDefined():
                primvars[name] = {'exists': False}
                continue
            values = primvar.Get()
            indices = primvar.GetIndices() if primvar.IsIndexed() else None
            value_count = len(values) if values is not None else 0
            index_count = len(indices) if indices is not None else None
            index_range_valid = None
            if indices is not None:
                index_range_valid = (len(indices) == expected_corners and
                                     all(0 <= int(index) < value_count for index in indices))
            else:
                index_range_valid = value_count == expected_corners
            primvars[name] = {
                'exists': True,
                'interpolation': str(primvar.GetInterpolation()),
                'element_size': int(primvar.GetElementSize()),
                'indexed': bool(primvar.IsIndexed()),
                'values_count': int(value_count),
                'indices_count': None if index_count is None else int(index_count),
                'expected_face_vertex_count': expected_corners,
                'encoding_count_matches_mesh': bool(index_range_valid),
            }
        record['mesh_geometry'] = {
            'type': 'Mesh',
            'face_count': int(len(face_counts)),
            'face_vertex_corner_count': expected_corners,
            'warning_primvars': primvars,
        }
        record['material'] = _bound_material_report(prim)
        results.append(record)
    return results


def _author_color_chart(stage, camera_path, width, height):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    camera_prim = stage.GetPrimAtPath(camera_path)
    camera = UsdGeom.Camera(camera_prim)
    if not camera:
        raise RuntimeError(f'RGB camera is not valid: {camera_path}')
    layer = Sdf.Layer.CreateAnonymous('rgb_color_chain_probe.usda')
    old_target = stage.GetEditTarget()
    session_layer = stage.GetSessionLayer()
    session_layer.subLayerPaths.insert(0, layer.identifier)
    try:
        stage.SetEditTarget(layer)
        chart_path = '/World/__CodexRgbColorChart'
        root = UsdGeom.Xform.Define(stage, chart_path)
        camera_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        camera_offset = Gf.Matrix4d(1.0)
        camera_offset.SetTranslate(Gf.Vec3d(0.0, 0.0, -CHART_DISTANCE))
        parent_world = UsdGeom.Xformable(root.GetPrim().GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        root.AddTransformOp().Set(Gf.Matrix4d(local_transform_from_world(
            camera_offset * camera_world, parent_world).tolist()))
        patches = []
        material_root = chart_path + '/Looks'
        rows = (('diffuse', 0.22), ('emissive', -0.22))
        for row_index, (mode, local_y) in enumerate(rows):
            for index, (name, rgb) in enumerate(COLOR_SWATCHES):
                emission_scale = 0.25 if name == 'neutral_gray' else 0.5
                diffuse, emission = color_patch_channels(rgb, mode, emission_scale)
                material_name = f'{mode}_{name}'
                material_path = f'{material_root}/{material_name}'
                material = UsdShade.Material.Define(stage, material_path)
                shader = UsdShade.Shader.Define(stage, material_path + '/Shader')
                shader.CreateIdAttr('UsdPreviewSurface')
                shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
                shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
                shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.0)
                shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')

                patch_path = f'{chart_path}/Patch_{row_index:02d}_{index:02d}_{material_name}'
                cube = UsdGeom.Cube.Define(stage, patch_path)
                cube.GetSizeAttr().Set(1.0)
                x = CHART_X_CENTERS[index]
                cube.AddTranslateOp().Set(Gf.Vec3d(x, local_y, 0.0))
                cube.AddScaleOp().Set(Gf.Vec3f(CHART_PATCH_SIZE[0], CHART_PATCH_SIZE[1], 0.025))
                UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)
                patches.append({
                    'name': name,
                    'material_mode': mode,
                    'input_rgb_linear': list(rgb),
                    'diffuse_rgb_linear': list(diffuse),
                    'emissive_rgb_linear': list(emission),
                    'local_xyz': [x, local_y, -CHART_DISTANCE],
                    'patch_size': list(CHART_PATCH_SIZE),
                    'roi': list(project_patch_roi(
                        local_xyz=(x, local_y, -CHART_DISTANCE), patch_size=CHART_PATCH_SIZE,
                        focal_length_mm=float(camera.GetFocalLengthAttr().Get()),
                        horizontal_aperture_mm=float(camera.GetHorizontalApertureAttr().Get()),
                        vertical_aperture_mm=float(camera.GetVerticalApertureAttr().Get()),
                        width=width, height=height, inner_fraction=0.5,
                    )),
                })
        return layer, old_target, patches
    except Exception:
        stage.SetEditTarget(old_target)
        if layer.identifier in session_layer.subLayerPaths:
            session_layer.subLayerPaths.remove(layer.identifier)
        raise


def _remove_layer(stage, layer, old_target):
    stage.SetEditTarget(old_target)
    paths = stage.GetSessionLayer().subLayerPaths
    if layer.identifier in paths:
        paths.remove(layer.identifier)


def _capture_rgb(app, preview, timeline, simulation_time):
    import numpy as np
    import omni.kit.app
    import omni.replicator.core as rep

    before = float(timeline.get_current_time())
    if abs(before - simulation_time) > 1e-8:
        raise RuntimeError(f'Frozen timeline mismatch: {before} != {simulation_time}')
    # Match the existing frozen replay barrier: drain stage/Hydra changes while
    # the timeline remains paused, then trigger exactly one explicit step call.
    for _ in range(3):
        omni.kit.app.get_app().update()
    if abs(float(timeline.get_current_time()) - before) > 1e-8:
        raise RuntimeError('App update advanced the frozen simulation timeline')
    if rep.orchestrator.get_status() == rep.orchestrator.Status.STOPPING:
        rep.orchestrator.wait_until_complete()
    status_before = str(rep.orchestrator.get_status())
    rep.orchestrator.step(rt_subframes=4, pause_timeline=True,
                          delta_time=0.0, wait_for_render=True)
    status_after = str(rep.orchestrator.get_status())
    if rep.orchestrator.get_status() != rep.orchestrator.Status.STEPPED:
        raise RuntimeError(f'RGB explicit render did not complete: {status_before} -> {status_after}')
    rgb = np.asarray(preview.rgb_annotator.get_data()).copy()
    if rgb.shape[:2] != (preview.height, preview.width):
        raise RuntimeError(f'Unexpected RGB annotator shape: {rgb.shape}')
    if rgb.dtype.name != 'uint8':
        raise RuntimeError(f'Expected uint8 LdrColor output, got {rgb.dtype}')
    return rgb, {
        'simulation_time_before': before,
        'simulation_time_after': float(timeline.get_current_time()),
        'explicit_render_status_before': status_before,
        'explicit_render_status_after': status_after,
        'rgb_render_completed_wall_s': time.time(),
        'rgb_annotator_shape': list(rgb.shape),
        'rgb_annotator_dtype': str(rgb.dtype),
    }


def _save_rgb(output, basename, rgb):
    import numpy as np
    from PIL import Image

    npy_path = output / f'{basename}.npy'
    png_path = output / f'{basename}.png'
    np.save(npy_path, rgb)
    Image.fromarray(rgb).save(png_path)
    decoded = np.asarray(Image.open(png_path))
    return png_path, compare_rgb_buffers(rgb, decoded)


def _latest_kit_log(prior_logs):
    if not KIT_LOG_ROOT.is_dir():
        return None
    candidates = sorted(KIT_LOG_ROOT.glob('kit_*.log'), key=lambda path: path.stat().st_mtime, reverse=True)
    new_logs = [path for path in candidates if str(path.resolve()) not in prior_logs]
    return (new_logs or candidates)[0] if candidates else None


def _read_uv_log(path):
    if path is None or not path.is_file():
        return [], None
    records = parse_uv_warnings(path.read_text(encoding='utf-8', errors='replace').splitlines())
    return records, str(path.resolve())


def _write_montage(output, samples, width, height):
    from PIL import Image, ImageDraw

    caption_height = 30
    montage = Image.new('RGB', (width * len(samples), height + caption_height), (24, 24, 24))
    draw = ImageDraw.Draw(montage)
    for index, sample in enumerate(samples):
        image = Image.open(sample['png_path']).convert('RGB')
        montage.paste(image, (index * width, caption_height))
        draw.text((index * width + 8, 8),
                  f"traj {sample['trajectory_index']} | t={sample['simulation_time_s']:.2f}s",
                  fill=(255, 255, 255))
    target = output / 'rgb_trajectory_montage.png'
    montage.save(target)
    return target


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        raise ValueError('--width and --height must be positive')
    config = load_scene_config(args.config)
    records = read_trajectory(args.trajectory)
    frame_indices = parse_frame_indices(args.frames, len(records))
    args.out.mkdir(parents=True, exist_ok=False)

    import hashlib
    source_hash = hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()
    config_hashes = {
        'scene_config_sha256': config.sha256,
        'thermal_config_sha256': hashlib.sha256(config.sensors.thermal_config.read_bytes()).hexdigest(),
        'sensor_config_sha256': hashlib.sha256(config.sensors.sensor_config.read_bytes()).hexdigest(),
    }
    prior_logs = {str(path.resolve()) for path in KIT_LOG_ROOT.glob('kit_*.log')} if KIT_LOG_ROOT.is_dir() else set()
    app = preview = controller = None
    failure = None
    report = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'renderer': 'PathTracing',
                             'samples_per_pixel_per_frame': 1, 'multi_gpu': False})
        from scripts.start_crazyflie_dual_camera_view import open_preview
        preview, controller = open_preview(
            app, scene_config=config, width=args.width, height=args.height,
            fast=False, start_controller=False, show_ui=False,
            renderer='PathTracing', disable_external_viewport=True,
        )
        import omni.timeline
        from pxr import Usd, UsdGeom, UsdPhysics
        from scripts.replay_pose import FrozenPoseRestorer

        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        timeline.set_auto_update(False)
        stage = preview.stage
        stage.SetEditTarget(stage.GetSessionLayer())
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        restorer = FrozenPoseRestorer(stage, config)

        first = records[frame_indices[0]]
        timeline.set_current_time(float(first['simulation_time']))
        timeline.commit()
        restorer.restore(first)
        chart_layer, previous_target, patches = _author_color_chart(
            stage, preview.mount.rgb_path, args.width, args.height)
        chart_rgb, chart_render_meta = _capture_rgb(app, preview, timeline,
                                                     float(first['simulation_time']))
        chart_png, chart_roundtrip = _save_rgb(args.out, 'color_chart_rgb', chart_rgb)
        chart_regions = []
        for patch in patches:
            chart_regions.append({
                **patch,
                'measured': region_statistics(chart_rgb, patch['roi']),
            })
        _remove_layer(stage, chart_layer, previous_target)

        sampled_frames = []
        for sequence_index, trajectory_index in enumerate(frame_indices):
            item = records[trajectory_index]
            sample_time = float(item['simulation_time'])
            timeline.set_current_time(sample_time)
            timeline.commit()
            desired_body, orientation = restorer.restore(item)
            rgb, render_meta = _capture_rgb(app, preview, timeline, sample_time)
            basename = f'frame_{sequence_index:03d}_trajectory_{trajectory_index:06d}_rgb'
            png_path, roundtrip = _save_rgb(args.out, basename, rgb)
            camera = UsdGeom.Xformable(stage.GetPrimAtPath(preview.mount.rgb_path))
            camera_world = camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            sample = {
                'sequence_index': sequence_index,
                'trajectory_index': trajectory_index,
                'trajectory_frame_id': item.get('frame_id'),
                'simulation_time_s': sample_time,
                'recorded_position': item.get('position'),
                'recorded_orientation_wxyz': orientation,
                'rgb_camera_world_matrix': [[float(camera_world[row][col]) for col in range(4)]
                                            for row in range(4)],
                'rgb_metrics': region_statistics(rgb, (0, 0, args.width, args.height)),
                'annotator_png_roundtrip': roundtrip,
                'render': render_meta,
                'png_path': str(png_path.resolve()),
                'npy_path': str((args.out / f'{basename}.npy').resolve()),
            }
            sampled_frames.append(sample)

        kit_log_path = args.kit_log if args.kit_log is not None else _latest_kit_log(prior_logs)
        uv_warnings, resolved_log_path = _read_uv_log(kit_log_path)
        rgb_camera = UsdGeom.Camera(stage.GetPrimAtPath(preview.mount.rgb_path))
        uv_meshes = _mesh_report(stage, uv_warnings, sampled_frames, rgb_camera,
                                 args.width, args.height, args.out / 'source_uv_inventory.json')
        all_uv_primvars_accounted = uv_warning_sources_inspected(uv_meshes)
        chart_mode_results = {}
        for mode in ('diffuse', 'emissive'):
            mode_colors = [row for row in chart_regions
                           if row['material_mode'] == mode and row['name'] != 'neutral_gray']
            mode_saturations = [row['measured']['mean_hsv_saturation'] for row in mode_colors]
            mode_lumas = [row['measured']['mean_luma'] for row in mode_colors]
            chart_mode_results[mode] = {
                'color_patch_count': len(mode_colors),
                'all_colors_visible': bool(mode_lumas and all(value > 2.0 for value in mode_lumas)),
                'mean_hsv_saturation': (sum(mode_saturations) / len(mode_saturations)
                                        if mode_saturations else None),
                'saturation_at_least_0_5': bool(mode_saturations and
                                                sum(mode_saturations) / len(mode_saturations) >= 0.5),
                'patches': mode_colors,
            }
        emissive_result = chart_mode_results['emissive']
        chart_color_visible = emissive_result['all_colors_visible']
        chart_is_colorful = bool(chart_color_visible and emissive_result['saturation_at_least_0_5'])
        roundtrips = [chart_roundtrip] + [sample['annotator_png_roundtrip'] for sample in sampled_frames]
        source_unchanged = hashlib.sha256(config.scene.stage.read_bytes()).hexdigest() == source_hash
        if not source_unchanged:
            raise RuntimeError('Root USD changed during the diagnostic')

        report = {
            'result': 'DIAGNOSTIC_COMPLETE',
            'diagnostic_only': True,
            'formal_data_eligible': False,
            'scene_config_path': str(config.source_path),
            'stage_path': str(config.scene.stage),
            'stage_sha256_before': source_hash,
            'stage_sha256_after': hashlib.sha256(config.scene.stage.read_bytes()).hexdigest(),
            'stage_unchanged': source_unchanged,
            **config_hashes,
            'trajectory_path': str(args.trajectory.resolve()),
            'requested_resolution': [args.width, args.height],
            'renderer': 'PathTracing',
            'samples_per_pixel_per_frame': 1,
            'rgb_camera_path': preview.mount.rgb_path,
            'temporary_chart': {
                'authored_in_anonymous_session_layer': True,
                'removed_after_capture': True,
                'capture': chart_render_meta,
                'annotator_png_roundtrip': chart_roundtrip,
                'saturated_color_swatches_visible': chart_is_colorful,
                'color_mode_results': chart_mode_results,
                'swatches': chart_regions,
                'interpretation': (
                    'Emissive saturated colors reach the LdrColor buffer and the decoded PNG preserves every byte, '
                    'so PNG saving does not desaturate this probe. Diffuse patches are measured separately under '
                    'the scene lighting to distinguish material/light response from the output encoding.'
                    if chart_is_colorful and chart_roundtrip['exact_equal'] else
                    'The emissive chart did not establish a clean colored-buffer baseline; inspect chart visibility, '
                    'renderer color management, and annotator output before drawing a scene-material conclusion.'
                ),
            },
            'scene_rgb_samples': sampled_frames,
            'all_rgb_png_roundtrips_exact': all(item['exact_equal'] for item in roundtrips),
            'trajectory_montage': str(_write_montage(args.out, sampled_frames,
                                                     args.width, args.height).resolve()),
            'uv_warning_diagnostic': {
                'source_uv_inventory_path': str((args.out / 'source_uv_inventory.json').resolve()),
                'kit_log_path': resolved_log_path,
                'warning_count': len(uv_warnings),
                'unique_mesh_count': len(uv_meshes),
                'all_warning_meshes_present_in_loaded_stage': all(
                    row['mesh_prim_valid_in_loaded_stage'] for row in uv_meshes) if uv_meshes else None,
                'all_warning_primvars_inspected': all_uv_primvars_accounted,
                'all_warning_source_signatures_accounted': all_uv_primvars_accounted,
                'direct_renderer_prototype_identity_verified': all(
                    row['mesh_prim_valid_in_loaded_stage'] for row in uv_meshes) if uv_meshes else None,
                'meshes': uv_meshes,
                'interpretation_limit': (
                    'A Hydra primvar-size warning alone does not prove visible color damage. Review each mesh '
                    'source, material/texture inputs, and indexed-primvar validity before changing any asset.'
                ),
            },
            'session_layer_only': True,
            'note': 'RGB-only diagnostic captures are not strict RGB/thermal sync evidence or calibrated color data.',
        }
        # This Kit build can terminate Python inside app.close(). Persist the
        # complete diagnostic and success message before invoking that boundary.
        (args.out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False),
                                              encoding='utf-8')
        print(f'RGB_COLOR_CHAIN_DIAGNOSTIC_COMPLETE {args.out}', flush=True)
    except Exception:
        failure = traceback.format_exc()
        (args.out / 'failure.txt').write_text(failure, encoding='utf-8')
    finally:
        if controller:
            controller.stop()
        if preview:
            try:
                preview.close()
            except Exception:
                if failure is None:
                    failure = traceback.format_exc()
        if app:
            try:
                app.close(exit_code=1 if failure else 0)
            except Exception:
                if failure is None:
                    failure = traceback.format_exc()
    if failure:
        (args.out / 'failure.txt').write_text(failure, encoding='utf-8')
        print(f'RGB_COLOR_CHAIN_DIAGNOSTIC_FAIL {args.out}', file=sys.stderr, flush=True)
        return 1
    print(f'RGB_COLOR_CHAIN_DIAGNOSTIC_COMPLETE {args.out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
