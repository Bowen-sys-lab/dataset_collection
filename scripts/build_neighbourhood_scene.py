"""Compose the neighbourhood street block into a runnable Isaac scene.

The source ``neighbourhood.usd`` is a Sketchfab-style asset pack with two
properties the rest of this project does not have:

* ``metersPerUnit = 0.01`` (centimetres), and
* ``upAxis = Y``.

USD does **not** rescale a referenced layer when its ``metersPerUnit`` differs
(verified: a 100-unit cube referenced into a metre stage still measures 100
units), and it never rotates between up axes.  So the conversion has to be
explicit: reference the source under ``/World/Site`` and give that Xform a
+90 deg rotation about X and a 0.01 scale.  After that the scene is metres and
Z-up like every other project scene.

The source also contains no robot, no physics scene and no lights, so this
overlay adds the Crazyflie reference used by the other scenes, a physics scene,
and a simple sun/sky pair for the RGB preview (the thermal model uses its own
weather file and solar model, not these lights).

Run with the Isaac Python launcher plus the USD library path; it is read-only
with respect to the source asset.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = ROOT / 'neighbourhood.usd'
OVERLAY = ROOT / 'neighbourhood' / 'neighbourhood_scene.usdc'
DRONE_SOURCE = ROOT / 'lesson_crazyflie.usd'

# category -> (target, infrared material, estimate key, bias K)
# The temperatures themselves are NOT chosen here: they come from
# validation/surface-temperatures-<scenario>.json, produced by
# scripts/estimate_surface_temperatures.py from the surface energy balance.  The bias only
# expresses a sub-surface effect the balance cannot see (a joint holds moisture and dirt).
CATEGORIES = {
    'wall': ('street_wall', 'brick_wall', 'wall', 0.0),
    'road': ('street_road', 'asphalt_dry', 'road', 0.0),
    'vegetation': ('street_vegetation', 'vegetation_leaf', 'vegetation', 0.0),
    'window': ('street_window', 'glass_windshield', 'window', 0.0),
    'ac': ('street_ac_unit', 'aircraft_aluminium_painted', None, 0.0),
    # Idle outdoor unit: an unpowered painted-metal box, so it follows the same surface
    # balance as the other painted metal in the scene rather than a made-up number.
    'ac_idle': ('street_ac_idle', 'aircraft_aluminium_painted', 'lamp', 0.0),
    'billboard': ('street_billboard', 'painted_composite', 'billboard', 0.0),
    'vehicle': ('street_vehicle', 'car_paint_black', 'vehicle_shaded', 0.0),
    # Bonnet over a recently stopped engine: an active residual source, not a solved surface.
    'car_engine': ('street_car_engine', 'car_paint_black', None, 0.0),
    'lamp': ('street_light_pole', 'aircraft_aluminium_painted', 'lamp', 0.0),
    'prop_plastic': ('street_traffic_plastic', 'abs_plastic_white', 'prop_plastic', 0.0),
    'prop_metal': ('street_traffic_metal', 'aircraft_aluminium_painted', 'prop_metal', 0.0),
    'other': ('street_ground', 'concrete', 'ground', 0.0),
    # Pavement under a tree canopy: shaded for most of the day, so it holds far less heat.
    # Three bands so the edge fades instead of ending on a hard circle.
    'shade_a': ('street_shade_a', 'asphalt_dry', 'road_shaded', 0.0),
    'shade_b': ('street_shade_b', 'asphalt_dry', 'road_shaded', 0.0),
    'shade_c': ('street_shade_c', 'asphalt_dry', 'road_shaded', 0.0),
    # Paving overlay: the paver faces sit a few millimetres proud of the joint, so the joint stays
    # damper and slightly cooler -- the effect that makes a tiled pavement legible in LWIR.
    'paving_tile': ('street_paving_tile', 'concrete', 'ground', 1.0),
    'paving_joint': ('street_paving_joint', 'concrete', 'ground', -3.0),
}

#: Paving overlay geometry: 60 cm pavers on a 20 mm joint, raised 6 mm above the joint plane.
TILE_PITCH_M = 0.3
TILE_GAP_M = 0.012
TILE_RAISE_M = 0.006
#: How far from a facade the paving band is assumed to run (no textures to read the real edge).
SIDEWALK_WIDTH_M = 3.0
#: OFF until the asset's textures/ folder is available.  Without the atlas there is no way to
#: tell paving from asphalt, and a guessed band would put tiles in the wrong places -- worse than
#: no tiles at all.  Set to True (and keep TILE_* as they are) once the atlas can be sampled by UV.
PAVING_ENABLED = True

#: A running split unit's casing, not a solved surface: it is an active heat source.
#: Literature range for the outdoor casing is 40-45 C at full load; a lightly loaded small
#: split unit sits lower, and 36 C keeps it inside the display range instead of clipping white.
AC_TEMPERATURE_C = 36.0
#: Bonnet of a car that stopped recently: the engine bay keeps the panel well above the body.
#: A no-wind saloon bonnet typically sits 20-30 K above ambient for the first half hour.
ENGINE_TEMPERATURE_C = 46.0
#: A lit street lamp runs its LED array and housing well above ambient (lens 50-60 C).
LAMP_ON_TEMPERATURE_C = 55.0
#: A tree shadow moves during the day, so the ground under it keeps only part of the
#: fully-shaded deficit.  0.55 of it is a reading, not a measurement.
SHADE_BANDS = {'shade_a': 0.30, 'shade_b': 0.55, 'shade_c': 0.85}
#: Sun at the golden-hour instant, used to project the canopy shade onto the ground.
SUN_ELEVATION_DEG = 15.7
SUN_AZIMUTH_DEG = 287.8
#: A UV spot inside the asphalt area of the road atlas, used by the shade patches so
#: the visible image shows asphalt rather than the default material.
ASPHALT_UV = (0.35, 0.30)
#: Not every unit is running, but no facade is dark either: within each building one unit in
#: AC_IDLE_EVERY is switched off and the rest run.
AC_IDLE_EVERY = 3

#: scenario -> (file suffix, display profile name, thermal environment, display window C, stage)
SCENARIOS = {
    'day': ('', 'neighbourhood_daytime',
            '../third_party/isaac-thermal-camera-simulation/configs/scenes/quad_flight_clear_noon.yaml',
            (20.0, 60.0), '../neighbourhood/neighbourhood_scene.usdc'),
    'dusk': ('_dusk', 'neighbourhood_dusk',
             'neighbourhood_dusk_environment.yaml',
             (18.0, 48.0), '../neighbourhood/neighbourhood_scene_dusk.usda'),
    'night': ('_night', 'neighbourhood_night',
              'model02_v4_night_environment.yaml', (10.0, 38.0),
              '../neighbourhood/neighbourhood_scene_night.usda'),
}

#: RGB preview lighting per scenario.  The thermal model never reads these lights -- they only
#: decide whether the visible image looks like noon, dusk or night.  Dusk keeps the sun just above
#: the horizon and warm, which is the condition dual-modal imagery is usually captured in.
LIGHTING = {
    'dusk': {
        # Golden hour: sun 15 deg above the horizon (rotateX = 90 - elevation), warm and still
        # strong.  The thermal side uses the same instant (start_utc 19:25 local).
        '/World/Lights/Sun': {'inputs:intensity': 4500.0, 'inputs:color': (1.0, 0.72, 0.45),
                              'xformOp:rotateY': -100.0, 'xformOp:rotateX': 75.0},
        '/World/Lights/Sky': {'inputs:intensity': 650.0, 'inputs:color': (0.62, 0.70, 0.92)},
    },
    'night': {
        '/World/Lights/Sun': {'inputs:intensity': 0.0},
        '/World/Lights/Sky': {'inputs:intensity': 450.0, 'inputs:color': (0.50, 0.60, 0.88)},
    },
}

#: Anything larger than this in any axis is a building mass, whatever the asset calls it.
#: The pack's "Window_*" and "AC_*" meshes measure 5-12 m across: they are facades, not fittings.
BUILDING_SIZE_M = 4.0

DRONE_TEMP = 35.0


def classify(path: str, max_dim_m: float = 0.0) -> str:
    """Semantic class: names first, then size for the asset's misleading facade names.

    Roads are metres across by nature and pots are 5 m plants, so size may only arbitrate
    the prims the pack calls "Wall", "Window" or "AC" -- those measure 5-12 m and are
    building masses, not fittings.
    """
    low = path.lower()
    if '__car_bonnet_off' in low:
        return 'vehicle'          # distractor cars: cold throughout, same as any parked car
    if '__car_engine' in low:
        return 'car_engine'
    if '__car_body' in low:
        return 'vehicle'
    for band, suffix in (('shade_a', '__shade_a'), ('shade_b', '__shade_b'), ('shade_c', '__shade_c')):
        if low.endswith(suffix):
            return band
    if 'paving_tile' in low:
        return 'paving_tile'
    if 'paving_joint' in low:
        return 'paving_joint'
    # Prims produced by the subset split carry their family as a suffix.  Air conditioners keep
    # one prim per unit, named <mesh>__ac__<material subset>.
    if '__ac__' in low:
        return 'ac'
    for suffix, category in (('__window', 'window'), ('__ac', 'ac'), ('__billboard', 'billboard'),
                             ('__wall', 'wall'), ('__plant', 'vegetation'), ('__other', 'other')):
        if low.endswith(suffix):
            return category
    if 'road' in low:
        return 'road'
    # Lamp must be tested before vegetation: "street" contains the substring "tree".
    if 'streetlight' in low or 'sm_light' in low:
        return 'lamp'
    if any(key in low for key in ('pot_', 'low_polyplant', 'acer', 'tree')):
        return 'vegetation'
    if any(key in low for key in ('bmw', 'e46')):
        return 'vehicle'
    if any(key in low for key in ('delineator', 'cone', 'drum', 'verticade', 'barricade')):
        return 'prop_plastic'
    if 'trafficbarrier' in low:
        return 'prop_metal'
    if max_dim_m > BUILDING_SIZE_M or 'wall' in low or 'window' in low or '_ac_' in low:
        return 'wall'
    return 'other'


def _mapped_bbox(prim, scale: float):
    """World box after the explicit Y-up/cm -> Z-up/m mapping: (x, -z, y) * scale."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                               UsdGeom.Tokens.proxy])
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if box.IsEmpty():
        return None
    lo, hi = box.GetMin(), box.GetMax()
    xs, ys, zs = [], [], []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                xs.append(x * scale)
                ys.append(-z * scale)
                zs.append(y * scale)
    return {'min': [min(xs), min(ys), min(zs)], 'max': [max(xs), max(ys), max(zs)]}


def _rect_distance(centre, box) -> float:
    """Planar distance from a point to an axis-aligned box footprint."""
    dx = max(box['min'][0] - centre[0], 0.0, centre[0] - box['max'][0])
    dy = max(box['min'][1] - centre[1], 0.0, centre[1] - box['max'][1])
    return math.hypot(dx, dy)


def _pick_start(boxes: dict, road_paths: list, target_xy=None) -> tuple[list, dict]:
    """Pick a take-off point on the roadway, far from the task target and clear of buildings.

    For a full-length trajectory the useful property is not "most open around the drone" but
    "as far from the target as the block allows, while still having room to fly".  Candidates are
    sampled every metre inside the road footprints and filtered to those with at least 2 m of
    clearance from anything standing; among those the one furthest from the target wins.
    """
    import numpy as np

    obstacles = [box for path, box in boxes.items()
                 if 'road' not in path.lower() and box['max'][2] > 0.4]
    best = None
    for path in road_paths:
        box = boxes[path]
        xs = np.arange(box['min'][0] + 0.5, box['max'][0] - 0.3, 0.5)
        ys = np.arange(box['min'][1] + 0.5, box['max'][1] - 0.3, 0.5)
        for x in xs:
            for y in ys:
                point = (float(x), float(y))
                clearance = min((_rect_distance(point, other) for other in obstacles), default=99.0)
                if clearance < 2.0:
                    continue
                if target_xy is None:
                    score = clearance
                else:
                    score = ((point[0] - target_xy[0]) ** 2 + (point[1] - target_xy[1]) ** 2) ** 0.5
                if best is None or score > best[0]:
                    best = (score, path, point, clearance)

    if best is None:                      # fall back to the old behaviour
        for path in road_paths:
            box = boxes[path]
            centre = [(box['min'][i] + box['max'][i]) / 2.0 for i in range(3)]
            best = (0.0, path, (centre[0], centre[1]), 0.0)
            break

    _, path, point, clearance = best
    height = 8.0
    distance = (round(((point[0] - target_xy[0]) ** 2 + (point[1] - target_xy[1]) ** 2) ** 0.5, 1)
                if target_xy is not None else None)
    evidence = {'road_prim': path,
                'planar_clearance_m': round(clearance, 2),
                'mount_height_m': height,
                'distance_to_target_m': distance,
                'centre': [round(point[0], 2), round(point[1], 2)]}
    return [point[0], point[1], height], evidence


def _set_light(light, attribute: str, value):
    """Set a UsdLux input across schema revisions (typed getter or attribute name)."""
    prim = light.GetPrim()
    attr = prim.GetAttribute(f'inputs:{attribute}')
    if not attr or not attr.IsValid():
        attr = prim.GetAttribute(attribute)
    if not attr or not attr.IsValid():
        raise RuntimeError(f'cannot author {attribute} on {prim.GetPath()}')
    attr.Set(value)


def _family(name: str) -> str:
    base = re.sub(r'(_\d+)+$', '', name).lower()
    if base.startswith('window'):
        return 'window'
    if base.startswith('wall'):
        return 'wall'
    if base.startswith('ac'):
        return 'ac'
    if 'plant' in base:
        return 'plant'
    if 'billboard' in base:
        return 'billboard'
    return 'other'


def _split_by_material_family(stage, root: str) -> dict:
    """Split multi-material building meshes into one prim per material family.

    The pack's four facade meshes carry ~167 GeomSubsets each (windows, sills, AC units,
    planters, billboards) under a single prim, so the thermal model -- which binds one
    temperature and one infrared material per prim -- collapses a whole building into one
    flat surface.  Splitting keeps the RGB result identical (every original material keeps
    its own subset, pointing at the same faces) while giving windows and AC units their own
    prim, and therefore their own temperature.
    """
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

    created = {}
    size_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in [p for p in stage.Traverse()
                 if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith(root)]:
        mesh = UsdGeom.Mesh(prim)
        subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
        if not subsets:
            continue
        # Only building masses are split.  A road or a prop keeps its single prim even when
        # its asset happens to carry subsets, so its class is never re-decided by a subset name.
        bound = size_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        max_dim = 0.0 if bound.IsEmpty() else max(bound.GetSize())
        if classify(prim.GetName(), max_dim) != 'wall':
            continue
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        face_start = np.concatenate(([0], np.cumsum(counts)))
        normals = mesh.GetNormalsAttr().Get()
        normals = None if normals is None else np.asarray(normals, dtype='float32')
        normal_interp = mesh.GetNormalsInterpolation()
        st_api = UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
        uvs = st_api.ComputeFlattened() if st_api else None
        uvs = None if uvs is None else np.asarray(uvs, dtype='float32')
        st_interp = st_api.GetInterpolation() if st_api else None

        def _interpolated(array, interpolation, corner, used, faces):
            if interpolation == 'faceVarying':
                return array[corner]
            if interpolation == 'vertex':
                return array[used]
            if interpolation == 'uniform':
                return array[faces]
            raise RuntimeError(f'{prim.GetPath()}: unsupported interpolation {interpolation}')

        groups = {}
        for sub in subsets:
            family = _family(sub.GetPrim().GetName())
            # Air conditioners are split per material subset, not merged per building: a whole
            # building's units under one prim can only be switched on or off together.
            key = (family, sub.GetPrim().GetName() if family == 'ac' else None)
            groups.setdefault(key, []).append(sub)
        parent = prim.GetParent().GetPath()
        for (family, unit), subs in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
            suffix = family if unit is None else f'{family}__{unit}'
            faces = sorted({int(i) for sub in subs for i in sub.GetIndicesAttr().Get()})
            if not faces:
                continue
            corner = np.concatenate([np.arange(face_start[f], face_start[f + 1]) for f in faces])
            used, remap = np.unique(indices[corner], return_inverse=True)
            dest = UsdGeom.Mesh.Define(stage, str(parent) + '/' + prim.GetName() + '__' + suffix)
            dest.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points[used].astype('float32')))
            dest.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts[faces].astype('int32')))
            dest.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(remap.astype('int32')))
            dest.CreateSubdivisionSchemeAttr(mesh.GetSubdivisionSchemeAttr().Get())
            dest.CreateDoubleSidedAttr(mesh.GetDoubleSidedAttr().Get())
            dest.CreateOrientationAttr(mesh.GetOrientationAttr().Get())
            if normals is not None:
                dest.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(
                    _interpolated(normals, normal_interp, corner, used, faces)))
                dest.SetNormalsInterpolation(normal_interp)
            if uvs is not None:
                st = UsdGeom.PrimvarsAPI(dest).CreatePrimvar(
                    'st', Sdf.ValueTypeNames.TexCoord2fArray, st_interp)
                st.Set(Vt.Vec2fArray.FromNumpy(
                    _interpolated(uvs, st_interp, corner, used, faces)))
            UsdGeom.Subset.SetFamilyType(dest, 'materialBind', UsdGeom.Tokens.partition)
            local = {f: i for i, f in enumerate(faces)}
            for sub in subs:
                new_faces = [local[int(f)] for f in sub.GetIndicesAttr().Get() if int(f) in local]
                new_sub = UsdGeom.Subset.CreateGeomSubset(
                    dest, sub.GetPrim().GetName(), UsdGeom.Tokens.face, new_faces, 'materialBind')
                material = UsdShade.MaterialBindingAPI(sub.GetPrim()).ComputeBoundMaterial()[0]
                if material:
                    UsdShade.MaterialBindingAPI.Apply(new_sub.GetPrim()).Bind(material)
            created[family] = created.get(family, 0) + 1
        stage.OverridePrim(prim.GetPath()).SetActive(False)
    return created


def _load_estimates(scenario: str) -> dict:
    """Surface temperatures must come from the energy-balance report, not from this file."""
    path = ROOT / 'validation' / f'surface-temperatures-{scenario}.json'
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} missing: run scripts/estimate_surface_temperatures.py --scenario {scenario} first')
    report = json.loads(path.read_text(encoding='utf-8'))
    temperatures = {name: row['surface_temperature_c'] for name, row in report['classes'].items()}
    needed = {key for _target, _material, key, _bias in CATEGORIES.values() if key}
    missing = needed - set(temperatures)
    if missing:
        raise KeyError(f'{path}: no estimate for {sorted(missing)}')
    return temperatures


def _build_paving(stage, source) -> list:
    """Give the asset's marble-tile paving squares a real paver relief.

    Now that textures/ is on disk the paving can be identified exactly: 25 flat 2 x 2 m meshes
    carry the marble_tiles material, and together they are the only paved ground in the block.
    Each square receives a 3 x 3 grid of pavers on a raised joint plane, so the visible image
    gets a shadow line at every joint and the thermal model can cool the joints separately.
    Asphalt, kerbs and lane markings are untouched because they are not these meshes.
    """
    import numpy as np
    from pxr import Usd, UsdGeom, UsdShade, Vt

    def texture_of(material):
        if not material:
            return None
        for shader in UsdShade.Material(material).GetPrim().GetChildren():
            if not shader.IsA(UsdShade.Shader):
                continue
            asset = UsdShade.Shader(shader).GetInput('file')
            if asset and asset.Get():
                return asset.Get().path.split('/')[-1]
        return None

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    joint_points, joint_indices = [], []
    tile_points, tile_indices = [], []
    patches = 0
    for prim in source.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        name = texture_of(UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0])
        if not name or 'marble_tiles' not in name.lower():
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        size = np.asarray(box.GetSize()) * 0.01
        if not (1.0 <= size[0] <= 3.0 and 1.0 <= size[2] <= 3.0 and size[1] < 0.2):
            continue
        lo, hi = np.asarray(box.GetMin()) * 0.01, np.asarray(box.GetMax()) * 0.01
        x0, x1 = lo[0], hi[0]
        y1, y0 = -lo[2], -hi[2]          # source Z maps to scene -Y
        if y0 > y1:
            y0, y1 = y1, y0
        # The marble squares sit ~8 mm below the asphalt, so lift the pavers to the road
        # surface first; otherwise they would be hidden under the road mesh.
        z = max(float(hi[1]), 0.0) + 0.002
        patches += 1
        base = len(joint_points)
        joint_points += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        joint_indices += [base, base + 1, base + 2, base, base + 2, base + 3]
        cells = 3
        cell_w = (x1 - x0) / cells
        cell_h = (y1 - y0) / cells
        for ix in range(cells):
            for iy in range(cells):
                gx = x0 + ix * cell_w + TILE_GAP_M / 2
                gy = y0 + iy * cell_h + TILE_GAP_M / 2
                tw, th = cell_w - TILE_GAP_M, cell_h - TILE_GAP_M
                zt = z + TILE_RAISE_M
                start = len(tile_points)
                tile_points += [(gx, gy, zt), (gx + tw, gy, zt),
                                (gx + tw, gy + th, zt), (gx, gy + th, zt)]
                tile_indices += [start, start + 1, start + 2, start, start + 2, start + 3]
    if not patches:
        return []

    created = []
    for name, points, indices in (('Paving_Joints', joint_points, joint_indices),
                                  ('Paving_Tiles', tile_points, tile_indices)):
        mesh = UsdGeom.Mesh.Define(stage, '/World/Paving/' + name)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype='float32')))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(indices) // 3)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        created.append(str(mesh.GetPath()))
    return created


def _add_night_lamps(wrapper_path, boxes: dict) -> int:
    """Give the street lamps a real light source in the night variant only.

    The asset's lamps are geometry with no emissive material, so a night render is lit by the sky
    alone and the lamps are invisible.  One warm sphere light is placed just below the top of each
    lamp head.  This lives in the night wrapper, so noon and dusk are untouched.
    """
    from pxr import Gf, Usd, UsdGeom, UsdLux

    stage = Usd.Stage.Open(str(wrapper_path))
    count = 0
    for path, box in boxes.items():
        size = max(box['max'][i] - box['min'][i] for i in range(3))
        if classify(path, size) != 'lamp':
            continue
        centre = [(box['min'][i] + box['max'][i]) / 2.0 for i in range(3)]
        centre[2] = box['max'][2] - 0.06
        light = UsdLux.SphereLight.Define(stage, f'/World/Lights/StreetLamp_{count:03d}')
        _set_light(light, 'intensity', 5000.0)
        _set_light(light, 'radius', 0.20)
        _set_light(light, 'color', Gf.Vec3f(1.0, 0.86, 0.62))
        transform = UsdGeom.Xformable(light)
        transform.ClearXformOpOrder()
        transform.AddTranslateOp().Set(Gf.Vec3d(*centre))
        count += 1
    stage.GetRootLayer().Save()
    return count


def _pick_task_target(stage, bindings, start, yaw_deg):
    """Pick the running outdoor unit nearest to the start view, and return its centre."""
    import math

    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    view = (math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)))
    best = None
    for binding in bindings:
        if binding['target'] != 'street_ac_unit':
            continue
        box = cache.ComputeWorldBound(stage.GetPrimAtPath(binding['prim_path'])).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        lo, hi = box.GetMin(), box.GetMax()
        centre = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
        vx, vy = centre[0] - start[0], centre[1] - start[1]
        distance = math.hypot(vx, vy)
        if distance < 1e-6:
            continue
        cosine = max(-1.0, min(1.0, (vx * view[0] + vy * view[1]) / distance))
        angle = math.degrees(math.acos(cosine))
        score = angle + 0.5 * distance
        if best is None or score < best[0]:
            best = (score, binding['prim_path'], centre)
    if best is None:
        raise RuntimeError('no running outdoor unit available as the task target')
    return best[1], best[2]


def _split_car_bonnet(stage, root: str) -> list:
    """Split the parked car into bonnet and body so the engine bay can be warmer.

    The imported car is a single mesh with no material subsets, so one temperature covers the
    whole vehicle and an engine that is still cooling cannot be represented.  The bonnet is
    recovered geometrically: faces that point up and sit in the front part of the body are the
    panel over the engine bay.  Both halves keep the original points and material, so the visible
    image is unchanged apart from the extra prim boundary.
    """
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, Vt

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    created = []
    for prim in [p for p in stage.Traverse()
                 if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith(root)]:
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        # Classify by full path: the car prim is named "Object_1" and only its ancestors say it
        # is a vehicle.
        if classify(str(prim.GetPath()), max(box.GetSize())) != 'vehicle':
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        world = np.c_[points, np.ones(len(points))] @ np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
        start_of = np.concatenate(([0], np.cumsum(counts)))

        lo, hi = np.asarray(box.GetMin()), np.asarray(box.GetMax())
        length_axis = int(np.argmax(hi - lo))
        front_limit = lo[length_axis] + 0.45 * (hi - lo)[length_axis]

        engine, body = [], []
        for face in range(len(counts)):
            corner = indices[start_of[face]:start_of[face + 1]]
            pts = world[corner][:, :3]
            centroid = pts.mean(axis=0)
            normal = np.cross(pts[min(1, len(pts) - 1)] - pts[0], pts[-1] - pts[0])
            norm = np.linalg.norm(normal)
            up = norm > 0 and (normal[2] / norm) > 0.5
            (engine if (up and centroid[length_axis] < front_limit) else body).append(face)
        if not engine or not body:
            continue

        normals = mesh.GetNormalsAttr().Get()
        normals = None if normals is None else np.asarray(normals, dtype='float32')
        normal_interp = mesh.GetNormalsInterpolation()
        parent = prim.GetParent().GetPath()
        for suffix, faces in (('car_engine', engine), ('car_body', body)):
            dest = UsdGeom.Mesh.Define(stage, f'{parent}/{prim.GetName()}__{suffix}')
            dest.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
            dest.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts[faces].astype('int32')))
            dest.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(
                np.concatenate([indices[start_of[f]:start_of[f + 1]] for f in faces]).astype('int32')))
            dest.CreateSubdivisionSchemeAttr(mesh.GetSubdivisionSchemeAttr().Get())
            dest.CreateDoubleSidedAttr(mesh.GetDoubleSidedAttr().Get())
            dest.CreateOrientationAttr(mesh.GetOrientationAttr().Get())
            if normals is not None:
                if normal_interp == 'faceVarying':
                    kept = np.concatenate([normals[start_of[f]:start_of[f + 1]] for f in faces])
                else:
                    kept = normals
                dest.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(kept, dtype='float32')))
                dest.SetNormalsInterpolation(normal_interp)
            created.append(str(dest.GetPath()))
        stage.OverridePrim(prim.GetPath()).SetActive(False)
    return created


def _park_distractor_cars(stage, composed, boxes: dict, start) -> list:
    """Park three cold copies of the saloon on the street as search distractors.

    One parked car makes the task "fly to the only car"; three more make it a search, and the
    target stays identifiable because it is the only one whose bonnet is still warm (the others
    are cold).  Copies reuse the target's own geometry (translated, same orientation), so they are
    the same model, same colour, same size -- identical in the visible image.
    """
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, Vt

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    body = engine = None
    for prim in composed.Traverse():
        path = str(prim.GetPath())
        if path.endswith('__car_body'):
            body = prim
        elif path.endswith('__car_engine'):
            engine = prim
    if body is None or engine is None:
        return []

    def world_points(prim):
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
        return (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]

    engine_points = world_points(engine)
    target_points = world_points(body)
    target_xy = target_points[:, :2].mean(axis=0)

    road_rects, obstacles = [], []
    for path, box in boxes.items():
        rect = (box['min'][0], box['min'][1], box['max'][0], box['max'][1])
        if 'road' in path.lower():
            road_rects.append(rect)
        elif box['max'][2] > 0.3:          # buildings, props, trees: anything standing up
            obstacles.append(rect)

    def distance(point, rect):
        x0, y0, x1, y1 = rect
        dx = max(x0 - point[0], 0.0, point[0] - x1)
        dy = max(y0 - point[1], 0.0, point[1] - y1)
        return float(np.hypot(dx, dy))

    lo_x = min(r[0] for r in road_rects)
    lo_y = min(r[1] for r in road_rects)
    hi_x = max(r[2] for r in road_rects)
    hi_y = max(r[3] for r in road_rects)

    spots = []
    for gx in np.arange(lo_x, hi_x, 1.5):
        for gy in np.arange(lo_y, hi_y, 1.5):
            point = np.array([gx, gy])
            if not any(r[0] <= point[0] <= r[2] and r[1] <= point[1] <= r[3] for r in road_rects):
                continue
            if min(distance(point, r) for r in obstacles) < 2.0:
                continue
            if np.linalg.norm(point - target_xy) < 10.0:
                continue
            if np.linalg.norm(point - np.array(start[:2])) < 8.0:
                continue
            if spots and min(np.linalg.norm(point - s) for s in spots) < 9.0:
                continue
            spots.append(point)
            if len(spots) == 3:
                break
        if len(spots) == 3:
            break
    if not spots:
        return []

    body_source = UsdGeom.Mesh(body)
    engine_source = UsdGeom.Mesh(engine)
    created = []
    for index, spot in enumerate(spots, start=2):
        offset = np.array([spot[0] - target_xy[0], spot[1] - target_xy[1], 0.0])
        for points, suffix, source in ((target_points + offset, 'car_body', body_source),
                                       (engine_points + offset, 'car_bonnet_off', engine_source)):
            path = f'/World/Parking/Car_{index:02d}__{suffix}'
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
            mesh.CreateFaceVertexCountsAttr(source.GetFaceVertexCountsAttr().Get())
            mesh.CreateFaceVertexIndicesAttr(source.GetFaceVertexIndicesAttr().Get())
            normals = source.GetNormalsAttr().Get()
            if normals is not None:
                mesh.CreateNormalsAttr(normals)
                mesh.SetNormalsInterpolation(source.GetNormalsInterpolation())
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDoubleSidedAttr(True)
            created.append(path)
    return created



def _hull2d(points):
    """Convex hull of a point cloud projected to XY (monotone chain, numpy only)."""
    import numpy as np

    pts = sorted(set(map(tuple, np.round(points, 2))))
    if len(pts) < 3:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def _build_shade_patches(stage, composed) -> list:
    """Project every canopy triangle onto the ground along the sun, so the shade looks like a tree.

    A convex hull gives a solid blob.  What a real tree casts is the silhouette of its own foliage:
    gaps between leaf clusters, a dense core and a ragged edge, stretched away from the sun by
    height / tan(elevation).  Each triangle of the canopy mesh is therefore projected individually;
    the union of those triangles is the silhouette, holes included.
    """
    import math

    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

    ground = 0.012
    tan_elevation = math.tan(math.radians(SUN_ELEVATION_DEG))
    # Direction the shadow travels: opposite the sun's azimuth.
    shadow = np.array([math.sin(math.radians(SUN_AZIMUTH_DEG + 180.0)),
                       -math.cos(math.radians(SUN_AZIMUTH_DEG + 180.0))])

    asphalt_material = None
    for prim in composed.Traverse():
        if prim.IsA(UsdGeom.Mesh) and 'road' in str(prim.GetPath()).lower():
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if bound:
                asphalt_material = bound
                break

    positions, indices = [], []
    trees = 0
    for prim in composed.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        name = str(prim.GetPath()).lower()
        if 'acer' not in name and 'pot_' not in name:
            continue
        mesh = UsdGeom.Mesh(prim)
        raw = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        faces = mesh.GetFaceVertexIndicesAttr().Get()
        if not raw or not counts:
            continue
        world = np.c_[np.asarray(raw, dtype=float), np.ones(len(raw))] @ np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
        trees += 1
        cursor = 0
        for count in counts:
            corners = list(faces[cursor:cursor + count])
            cursor += count
            for k in range(1, count - 1):
                triangle = world[[corners[0], corners[k], corners[k + 1]]][:, :3]
                if float(triangle[:, 2].mean()) < 1.0:
                    continue                      # trunk and base: no canopy shadow
                base = len(positions)
                for point in triangle:
                    offset = shadow * ((float(point[2]) - ground) / tan_elevation)
                    positions.append((float(point[0] + offset[0]),
                                      float(point[1] + offset[1]), ground))
                indices += [base, base + 1, base + 2]
    # Everything else that stands above the ground casts a shadow too: buildings, cars, lamp
    # posts, cones, barriers.  Their silhouette is a box, so the eight corners are projected and
    # hulled -- cheap and correct for blocky objects, unlike a per-triangle sweep of a facade.
    ground_meshes = ('road', 'paving', 'canopy', 'shade')
    index = 0
    for prim in composed.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        path_low = str(prim.GetPath()).lower()
        if path_low.startswith('/world/crazyflie') or path_low.startswith('/world/lights'):
            continue
        if any(token in path_low for token in ground_meshes):
            continue
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        lo, hi = np.asarray(box.GetMin(), dtype=float), np.asarray(box.GetMax(), dtype=float)
        if hi[2] < 0.15:
            continue                          # flat on the ground: casts nothing
        corners = []
        for dx in (lo[0], hi[0]):
            for dy in (lo[1], hi[1]):
                for dz in (lo[2], hi[2]):
                    offset = shadow * ((dz - ground) / tan_elevation)
                    corners.append((dx + offset[0], dy + offset[1]))
        hull = _hull2d(np.asarray(corners))
        if len(hull) < 3:
            continue
        index += 1
        layer = ground + 0.0002 * (index % 40)
        centre = hull.mean(axis=0)
        base = len(positions)
        positions.append((float(centre[0]), float(centre[1]), layer))
        for point in hull:
            positions.append((float(point[0]), float(point[1]), layer))
        for k in range(len(hull)):
            indices += [base, base + 1 + k, base + 1 + (k + 1) % len(hull)]

    if not positions:
        return []

    mesh = UsdGeom.Mesh.Define(stage, '/World/Shade/CanopyShade__shade_b')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(positions, dtype='float32')))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(indices) // 3)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        'st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([(ASPHALT_UV[0], ASPHALT_UV[1])] * len(positions)))
    if asphalt_material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(asphalt_material)
    return [str(mesh.GetPath())]


def build():
    import yaml
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    if OVERLAY.exists():
        raise FileExistsError(f'Overlay already exists: {OVERLAY}')

    source = Usd.Stage.Open(str(SOURCE))
    if source is None:
        raise RuntimeError(f'Cannot open {SOURCE}')
    scale = float(UsdGeom.GetStageMetersPerUnit(source))
    up_axis = str(UsdGeom.GetStageUpAxis(source))
    if up_axis != 'Y' or abs(scale - 0.01) > 1e-9:
        raise RuntimeError(f'Unexpected source conventions: up={up_axis} metersPerUnit={scale}')

    meshes = [prim for prim in source.Traverse() if prim.IsA(UsdGeom.Gprim)]
    boxes = {}
    for prim in meshes:
        box = _mapped_bbox(prim, scale)
        if box is not None:
            boxes[str(prim.GetPath())] = box
    if not boxes:
        raise RuntimeError('No renderable geometry under the source stage')

    road_paths = [path for path in boxes if 'road' in path.lower()]
    car_boxes = [box for path, box in boxes.items()
                 if 'bmw' in path.lower() or 'e46' in path.lower()]
    car_xy = (((car_boxes[0]['min'][0] + car_boxes[0]['max'][0]) / 2.0,
               (car_boxes[0]['min'][1] + car_boxes[0]['max'][1]) / 2.0)
              if car_boxes else None)
    start, start_evidence = _pick_start(boxes, road_paths, car_xy)

    walls = [boxes[path] for path in boxes
             if classify(path, max(boxes[path]['max'][i] - boxes[path]['min'][i]
                                   for i in range(3))) == 'wall']
    if walls:
        goal = [sum((box['min'][i] + box['max'][i]) / 2 for box in walls) / len(walls)
                for i in range(3)]
    else:
        goal = [0.0, 0.0, 2.0]
    yaw = math.degrees(math.atan2(goal[1] - start[1], goal[0] - start[0]))

    OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(OVERLAY))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetTimeCodesPerSecond(60)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1000000)

    site = UsdGeom.Xform.Define(stage, '/World/Site')
    site.GetPrim().GetReferences().AddReference('../neighbourhood.usd', '/World')
    site_xf = UsdGeom.Xformable(site)
    site_xf.ClearXformOpOrder()
    site_xf.AddRotateXOp().Set(90.0)
    site_xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))

    # Reuse the project's Crazyflie subtree; the rest of the lesson stage is not pulled in.
    drone = stage.DefinePrim('/World/Crazyflie', 'Xform')
    drone.GetReferences().AddReference('../lesson_crazyflie.usd', '/World/Crazyflie')
    drone_xf = UsdGeom.Xformable(drone)
    drone_xf.ClearXformOpOrder()
    drone_xf.AddTranslateOp().Set(Gf.Vec3d(*start))
    drone_xf.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw).GetQuat()))
    drone_xf.AddScaleOp().Set(Gf.Vec3f(1))

    # Forward view with a 17 deg downward tilt: the useful targets (AC units, windows) are on walls.
    # Both cameras share the same install matrix so the RGB and thermal views are aligned by
    # construction; the launcher still re-aligns them at runtime (align_thermal_to_rgb).
    camera_path = '/World/Crazyflie/body/Camera_mount/Camera'
    camera_matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(0, 0, 0.02), Gf.Vec3d(1, 0, -0.30), Gf.Vec3d(0, 0, 1)).GetInverse()
    camera = UsdGeom.Camera(stage.OverridePrim(camera_path))
    camera.GetPrim().SetTypeName('Camera')
    camera.MakeMatrixXform().Set(camera_matrix)
    camera.CreateFocalLengthAttr(18.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(16.764)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000.0))

    # Thermal camera of the dual-camera rig.  Camera_mount is identity in the robot asset, so an
    # identity Sensors container plus the same local matrix puts both cameras at one world pose.
    sensors_path = '/World/Crazyflie/body/Sensors'
    thermal_path = sensors_path + '/ThermalCamera'
    sensors = UsdGeom.Xform.Define(stage, sensors_path)
    UsdGeom.Xformable(sensors).ClearXformOpOrder()
    thermal_camera = UsdGeom.Camera.Define(stage, thermal_path)
    thermal_camera.MakeMatrixXform().Set(camera_matrix)
    thermal_camera.CreateFocalLengthAttr(18.0)
    thermal_camera.CreateHorizontalApertureAttr(20.955)
    thermal_camera.CreateVerticalApertureAttr(16.764)
    thermal_camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000.0))

    # The lesson asset still carries a stale generated preview camera group.  The launcher hides
    # and redirects it at runtime; deactivating it here keeps the saved scene to exactly two
    # cameras so it is also correct when opened by hand.
    stale_preview = '/World/Crazyflie/body/Sensors_preview'
    if stage.GetPrimAtPath(stale_preview):
        stage.OverridePrim(stale_preview).SetActive(False)

    physics = UsdPhysics.Scene.Define(stage, '/PhysicsScene')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)

    # RGB preview lighting only; the thermal model uses its own weather and solar model.
    dome = UsdLux.DomeLight.Define(stage, '/World/Lights/Sky')
    _set_light(dome, 'intensity', 1500.0)
    sun = UsdLux.DistantLight.Define(stage, '/World/Lights/Sun')
    _set_light(sun, 'intensity', 9000.0)
    _set_light(sun, 'angle', 0.53)
    sun_xf = UsdGeom.Xformable(sun)
    sun_xf.ClearXformOpOrder()
    sun_xf.AddRotateYOp().Set(-35.0)
    sun_xf.AddRotateXOp().Set(55.0)

    stage.GetRootLayer().Save()

    # Split the facade meshes so windows and AC units can carry their own temperature.
    reopened = Usd.Stage.Open(str(OVERLAY))
    split_counts = _split_by_material_family(reopened, '/World/Site')
    _split_car_bonnet(reopened, '/World/Site')
    parked_cars = _park_distractor_cars(reopened, reopened, boxes, start)
    shade_paths = _build_shade_patches(reopened, reopened)
    paving_paths = _build_paving(reopened, source) if PAVING_ENABLED else []
    reopened.GetRootLayer().Save()

    # Per-scenario RGB lighting.  A wrapper stage sublayers the shared geometry and only
    # overrides the two lights, so dusk/night do not duplicate the 19 MB overlay.
    def _write_lighting_variant(scenario: str, overrides: dict) -> None:
        wrapper = ROOT / 'neighbourhood' / f'neighbourhood_scene_{scenario}.usda'
        variant = Usd.Stage.CreateNew(str(wrapper))
        variant.GetRootLayer().subLayerPaths.append(OVERLAY.name)
        UsdGeom.SetStageUpAxis(variant, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(variant, 1.0)
        variant.SetTimeCodesPerSecond(60)
        variant.SetStartTimeCode(0)
        variant.SetEndTimeCode(1000000)
        variant.SetDefaultPrim(variant.OverridePrim('/World'))
        for prim_path, attributes in overrides.items():
            prim = variant.OverridePrim(prim_path)
            for attribute_name, value in attributes.items():
                attribute = prim.GetAttribute(attribute_name)
                if not attribute or not attribute.IsValid():
                    raise RuntimeError(f'{prim_path}: no attribute {attribute_name}')
                attribute.Set(value)
        variant.GetRootLayer().Save()

    for scenario_name, overrides in LIGHTING.items():
        _write_lighting_variant(scenario_name, overrides)
        if scenario_name == 'night':
            _add_night_lamps(ROOT / 'neighbourhood' / 'neighbourhood_scene_night.usda', boxes)

    # ---- generated configuration ------------------------------------------------
    # Build the bindings from the *composed* stage: the reference introduces its own
    # namespace prefix, so the source paths are not the runtime paths.
    composed = Usd.Stage.Open(str(OVERLAY))
    site_paths = sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange(composed.GetPrimAtPath('/World/Site'), Usd.TraverseInstanceProxies())
        if prim.IsA(UsdGeom.Gprim))
    if not site_paths:
        raise RuntimeError('no renderable geometry under /World/Site after composition')
    paving_root = composed.GetPrimAtPath('/World/Paving')
    if paving_root:
        site_paths += [str(prim.GetPath()) for prim in Usd.PrimRange(paving_root)
                       if prim.IsA(UsdGeom.Gprim)]
    shade_root = composed.GetPrimAtPath('/World/Shade')
    if shade_root:
        site_paths += [str(prim.GetPath()) for prim in Usd.PrimRange(shade_root)
                       if prim.IsA(UsdGeom.Gprim)]
    parking_root = composed.GetPrimAtPath('/World/Parking')
    if parking_root:
        site_paths += [str(prim.GetPath()) for prim in Usd.PrimRange(parking_root)
                       if prim.IsA(UsdGeom.Gprim)]
    site_paths = sorted(set(site_paths))

    bindings, counts = [], {}
    ac_per_building = {}
    size_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for index, path in enumerate(site_paths, start=1):
        bound = size_cache.ComputeWorldBound(composed.GetPrimAtPath(path)).ComputeAlignedRange()
        max_dim = 0.0 if bound.IsEmpty() else max(bound.GetSize())
        category = classify(path, max_dim)
        if category == 'ac':
            # Count per parent (one parent per building mesh) so every building keeps most of
            # its units running and switches a few off, instead of a whole facade going dark.
            building = path.rsplit('/', 1)[0]
            seen = ac_per_building.get(building, 0)
            ac_per_building[building] = seen + 1
            if (seen + 1) % AC_IDLE_EVERY == 0:
                category = 'ac_idle'
        target, material, _key, _bias = CATEGORIES[category]
        counts[category] = counts.get(category, 0) + 1
        bindings.append({'name': f'{category}_{index:03d}', 'prim_path': path,
                         'include_children': False, 'target': target, 'material': material})

    bindings.append({'name': 'drone', 'prim_path': '/World/Crazyflie',
                     'include_children': True, 'target': 'street_drone',
                     'material': 'painted_composite'})

    estimates = {name: _load_estimates(name) for name in SCENARIOS}

    # ---- task (VLN) -------------------------------------------------------------
    # Dusk, one parked car.  The car body has cooled faster than the pavement, so in the thermal
    # image it reads as a cold patch on warm ground while the visible image shows an ordinary
    # parked saloon.  That is what makes the task dual-modal instead of a colour search.
    # The target is the original car under /World/Site -- the only one with a warm bonnet; the
    # copies live under /World/Parking and must never be picked as the task target.
    car = next((b for b in bindings
                if b['target'] == 'street_vehicle' and b['prim_path'].startswith('/World/Site/')), None)
    if car is None:
        raise RuntimeError('no vehicle binding to use as the task target')
    car_box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        composed.GetPrimAtPath(car['prim_path'])).ComputeAlignedRange()
    lo, hi = car_box.GetMin(), car_box.GetMax()
    target_centre = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
    task = {
        'task_id': 'neighbourhood_find_warm_car_01',
        'instruction': (
            '街区内停着几辆车，请找到刚刚熄火的那一辆。'
            '可见光画面里这几辆车同款同色、外观完全一致，无法区分；'
            '热红外画面里，只有目标的引擎盖是亮的，其余车辆的引擎盖都与车身一样冷，'
            '而且目标紧挨着一道交通护栏停放。'
            '请交替查看两路画面确认目标，不要只依赖可见光。'
            '接近到 5 米以内悬停 3 秒以上确认后，按 T 标记完成。'),
        'target_prim_path': car['prim_path'],
        'target_description': (
            '街边停放的浅灰色四门轿车（BMW E46 模型，长约 4.5 米），刚熄火不久：'
            '引擎盖温度明显高于车身其余部分，车旁 1.5 米内有一道红白交通护栏。'
            '场景中另有 3 辆同款同色车辆作为干扰，它们的引擎盖都是冷的。'),
        'environment_description': (
            '傍晚的临街住宅街区：砖砌多层住宅、行道树、路灯、交通锥桶与围挡。'
            '路面与墙体仍保有白天余热，静止车辆已冷却，因此热像中车身比地面暗、只有目标引擎盖偏亮。'),
        'environment_landmarks': ['砖砌住宅楼', '临街行道树', '路灯杆', '交通锥桶与围挡', '红白交通护栏', '沥青路面'],
        'start_position': [round(value, 2) for value in start],
        'target_position': [round(value, 2) for value in target_centre],
        'target_position_semantics': '世界坐标系（米），车辆网格包围盒中心',
        'notes': (
            '双模态判别任务：目标与干扰车在可见光下无差别，只有热像能区分（引擎盖余温）。'
            '判定方式：人工按 T 标记 + 自动核查（距目标 ≤5 m、目标在画面内、悬停 ≥3 s），'
            '评估器 scripts/evaluate_neighbourhood_task.py。'),
    }
    task_path_out = ROOT / 'configs' / 'neighbourhood_task.yaml'
    task_path_out.write_text(
        yaml.safe_dump(task, allow_unicode=True, sort_keys=False, width=110), encoding='utf-8')

    written = {}
    for scenario, (suffix, profile_name, thermal_scene, window, stage_path) in SCENARIOS.items():
        temperatures = estimates[scenario]
        targets = {}
        for category, (target, _material, key, bias) in CATEGORIES.items():
            if category == 'car_engine':
                celsius = ENGINE_TEMPERATURE_C
            elif category == 'lamp' and scenario in ('dusk', 'night'):
                celsius = LAMP_ON_TEMPERATURE_C
            elif category in SHADE_BANDS:
                sunlit, shaded = temperatures['road'], temperatures['road_shaded']
                celsius = sunlit + SHADE_BANDS[category] * (shaded - sunlit)
            elif key is None:
                celsius = AC_TEMPERATURE_C
            else:
                celsius = temperatures[key] + bias
            targets[target] = {'mode': 'fixed', 'temperature_c': celsius}
        targets['street_drone'] = {'mode': 'fixed', 'temperature_c': DRONE_TEMP}

        thermal = {
            'schema_version': 1,
            'defaults': {'target': 'street_ground', 'material': 'asphalt_dry'},
            'targets': targets,
            'bindings': bindings,
        }
        config = {
            'schema_version': 1,
            'scene': {
                'stage': stage_path,
                'thermal_scene': thermal_scene,
            },
            'drone': {
                'prim_path': '/World/Crazyflie',
                'body_path': '/World/Crazyflie/body',
                'camera_path': camera_path,
                'minimum_height': 1.5,
                'horizontal_speed': 2.0,
            },
            'sensors': {
                'width': 160,
                'height': 128,
                'sensor_config': '../third_party/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml',
                'thermal_config': f'neighbourhood_thermal{suffix}.yaml',
                'preview_fast': True,
                'align_thermal_to_rgb': True,
                'flat_field_enabled': True,
                'sensor_seed': 0,
                'display_config': f'neighbourhood_display{suffix}.yaml',
                'material_directory': 'materials',
            },
        }
        display = {
            'schema_version': 1,
            'name': profile_name,
            'mode': 'fixed_apparent_temperature',
            'temperature_min_c': window[0],
            'temperature_max_c': window[1],
            'dde_gain': 0.0,
            'polarity': 'white_hot',
            'palette': 'gray',
        }
        # The car task is defined for the dusk variant only (its thermal contrast does not exist
        # at noon or at night).
        if suffix == '_dusk':
            config['task'] = {'file': 'neighbourhood_task.yaml'}
        for name, payload in ((f'neighbourhood_scene{suffix}.yaml', config),
                              (f'neighbourhood_thermal{suffix}.yaml', thermal),
                              (f'neighbourhood_display{suffix}.yaml', display)):
            path = ROOT / 'configs' / name
            if path.exists():
                raise FileExistsError(path)
            path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False,
                                           default_flow_style=None, width=200), encoding='utf-8')
            written[name] = str(path)

    summary = {
        'overlay': str(OVERLAY),
        'source_units': {'meters_per_unit': scale, 'up_axis': up_axis,
                         'conversion': 'rotate +90 X, scale 0.01'},
        'geometry_bound': len(site_paths),
        'source_geometry': len(boxes),
        'split_surfaces': split_counts,
        'category_counts': counts,
        'start_position_m': [round(value, 2) for value in start],
        'start_yaw_deg': round(yaw, 1),
        'start_evidence': start_evidence,
        'ground_z_m': round(max((boxes[p]['max'][2] for p in road_paths), default=0.0), 3),
        'estimated_surface_temperatures_c': estimates,
        'ac_assumed_c': AC_TEMPERATURE_C,
        'configs': written,
    }
    (OVERLAY.parent / 'build_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
