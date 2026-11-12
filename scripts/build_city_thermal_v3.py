"""Build a traceable City material-partition overlay and versioned thermal inputs."""
import sys, json, hashlib, collections, shutil
from pathlib import Path
import yaml
sys.path.insert(0, 'D:/isaacsim/extscache/omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312')
from pxr import Usd, UsdGeom, UsdShade, Sdf
from scripts.city_material_partition import partition_mesh
from scripts.thermal_scene_config import renderable_paths

ROOT = Path(__file__).resolve().parents[1]


def classify(material):
    name = material.GetPrim().GetName().lower() if material else ''
    if 'window' in name: return 'glass'
    if 'bark' in name: return 'bark'
    if any(s in name for s in ('plant', 'cluster', 'leaf')): return 'foliage'
    if 'modular_road' in name: return 'asphalt'
    if 'wall' in name: return 'masonry'
    if name == 'ac' or name.startswith('ac_'): return 'background_ac'
    if 'billboard' in name: return 'signboard'
    if 'solarpanel' in name: return 'glass'
    return 'painted_metal' if 'streetlights' in name else 'unknown'


def build():
    source = ROOT/'city_200m_v2_flight.usda'
    output = ROOT/'city_200m_v3_flight.usdc'
    if output.exists(): raise FileExistsError(output)
    layer = Sdf.Layer.CreateNew(str(output))
    layer.subLayerPaths = ['./city_200m_v2_flight.usda']
    stage = Usd.Stage.Open(layer)
    source_stage=Usd.Stage.Open(str(source))
    for key in ('defaultPrim','metersPerUnit','upAxis','startTimeCode','endTimeCode','timeCodesPerSecond','framesPerSecond'):
        stage.SetMetadata(key,source_stage.GetMetadata(key))
    original = yaml.safe_load((ROOT/'configs/city_200m_v2_thermal.yaml').read_text(encoding='utf-8'))
    cfg = json.loads(json.dumps(original))
    values = {'asphalt':29, 'bark':25, 'foliage':23, 'glass':22, 'masonry':27,
              'painted_metal':25, 'unknown':26, 'equipment_case':28, 'pavement':27,
              'background_ac':30, 'signboard':25}
    for name, temperature in values.items():
        cfg['targets'][name]={'mode':'fixed','temperature_c':temperature}
    categories = {'glass':'glass_windshield','bark':'wood_pine','foliage':'vegetation_leaf',
                  'asphalt':'asphalt_dry','masonry':'concrete_weathered',
                  'background_ac':'metal_sheet_painted_grey','signboard':'metal_sheet_painted_grey',
                  'painted_metal':'car_paint_white','unknown':'concrete_weathered'}
    meshes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)
              and str(p.GetPath()).startswith('/World/Blocks/')]
    derived=[]
    for path in meshes:
        parts=partition_mesh(stage,path,classify)
        derived.extend(parts)
        if parts: print(f'[partition] {path.rsplit("/",1)[-1]} -> {len(parts)}',flush=True)
    hidden={p['source'] for p in derived}
    cfg['bindings']=[b for b in cfg['bindings'] if b['prim_path'] not in hidden]
    for part in derived:
        category=part['category']
        cfg['bindings'].append(dict(name=f'v3_partition_{len(cfg["bindings"]):04d}',prim_path=part['path'],
                                    include_children=False,target=category,material=categories[category]))
    # Single-material background assets can be classified without touching geometry.
    for binding in cfg['bindings']:
        if binding['target']!='unknown' or not binding['prim_path'].startswith('/World/Blocks/'):continue
        prim=stage.GetPrimAtPath(binding['prim_path'])
        mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        category=classify(mat)
        if category!='unknown':binding.update(target=category,material=categories[category])
    layer.customLayerData={'thermalRevision':'city_material_partition_v3',
                           'sourceStageSha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    layer.Save()
    def write(name,data):
        (ROOT/'configs'/name).write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True),encoding='utf-8')
    write('city_200m_v3_thermal.yaml',cfg)
    display=dict(schema_version=1,name='city_200m_v3_night_materials',mode='fixed_apparent_temperature',
                 temperature_min_c=18.0,temperature_max_c=45.0,dde_gain=0.0,polarity='white_hot',palette='gray')
    write('city_200m_v3_display.yaml',display)
    scene=yaml.safe_load((ROOT/'configs/city_200m_v2_scene.yaml').read_text(encoding='utf-8'))
    scene['scene']['stage']='../city_200m_v3_flight.usdc'
    material_dir=ROOT/'configs/materials/city_200m_v3'
    shutil.copytree(ROOT/'configs/materials/city_200m_v2',material_dir)
    scene['sensors'].update(thermal_config='city_200m_v3_thermal.yaml',display_config='city_200m_v3_display.yaml',material_directory='materials/city_200m_v3')
    write('city_200m_v3_scene.yaml',scene)
    actual=set(renderable_paths(stage))
    bindings={b['prim_path'] for b in cfg['bindings']}
    report=dict(version='city_material_partition_v3',source_stage=str(source),
        source_stage_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        overlay_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        partitioned_original_meshes=len(hidden),derived_meshes=len(derived),
        derived_faces=sum(p['faces'] for p in derived),category_meshes=dict(collections.Counter(p['category'] for p in derived)),
        unmapped_visible=[p for p in actual-bindings if not p.startswith('/World/CityFlight/')],
        unused_bindings=sorted(bindings-actual),partitions=derived,
        qualification='Synthetic prescribed night materials; not energy balance or calibrated temperatures; hot cabinet may saturate display only.')
    (ROOT/'validation/diagnostics/city-thermal-v3-20261108/build_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='partitions'},indent=2))


if __name__=='__main__':build()
