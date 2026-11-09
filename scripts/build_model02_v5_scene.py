"""Compose Blender-authored ground over v4 without changing its thermal bindings."""
import copy,hashlib,json,os
from pathlib import Path
from pxr import Usd,UsdGeom,UsdShade

ROOT=Path('D:/IsaacSim_Demo');OUT=ROOT/'factory/model02_thermal_v5';OLD=ROOT/'factory/model02_thermal_v4'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    source=json.loads((OLD/'isaac_manifest.json').read_text(encoding='utf-8'))
    lookup={r['surface_id']:r for r in source['surfaces']}
    report=json.loads((OUT/'blender_ground_report.json').read_text())
    exported=Usd.Stage.Open(str(OUT/'model02_v5_ground.usdc'))
    paths={};maps=[]
    for p in exported.Traverse():
        if p.IsA(UsdGeom.Mesh):
            parent=p
            while parent and not parent.GetAttribute('userProperties:surface_id').HasAuthoredValueOpinion():parent=parent.GetParent()
            assert parent,('No stable surface ID',str(p.GetPath()))
            sid=parent.GetAttribute('userProperties:surface_id').Get()
            assert str(p.GetPath())==lookup[sid]['usd_prim_path'],('Path changed',sid,str(p.GetPath()),lookup[sid]['usd_prim_path'])
            paths[sid]=str(p.GetPath())
            material,_=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
            assert material and 'V5_' in str(material.GetPath()),('Missing new material',sid)
        if p.IsA(UsdShade.Shader):
            shader=UsdShade.Shader(p)
            if shader.GetIdAttr().Get()=='UsdUVTexture':
                asset=shader.GetInput('file').Get();assert asset.resolvedPath and Path(asset.resolvedPath).is_file()
                maps.append(asset.resolvedPath)
    assert set(paths)==set(report['modified_names'])
    assert len(set(maps))==12,('PBR map set incomplete',maps)
    output=OUT/'model02_v5_scene.usda';assert not output.exists()
    stage=Usd.Stage.CreateNew(str(output));stage.GetRootLayer().subLayerPaths=['./model02_v5_ground.usdc','../model02_thermal_v4/model02_v4_scene.usda']
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    stage.SetTimeCodesPerSecond(60);stage.SetStartTimeCode(0);stage.SetEndTimeCode(1000000)
    oldstage=Usd.Stage.Open(str(OLD/'model02_v4_scene.usda'))
    profile=copy.deepcopy(dict(oldstage.GetRootLayer().customLayerData['uavLighting']))
    profile['dependencies']={os.path.relpath((OLD/p).resolve(),OUT).replace('\\','/'):h for p,h in profile['dependencies'].items()}
    for p in [OLD/'model02_v4_scene.usda',OUT/'model02_v5_ground.usdc',*[Path(p) for p in sorted(set(maps))]]:
        profile['dependencies'][os.path.relpath(p,OUT).replace('\\','/')]=sha(p)
    stage.GetRootLayer().customLayerData={'uavLighting':profile,'v5Ground':{
        'source_blend_sha256':report['configured_blend_sha256'],
        'assumptions':'2mm filled brick joints,3mm edge chamfers;1mm concrete control joint; grass soft normal relief; no emphasized grain',
        'thermal_configuration':'../model02_thermal_v4/configuration/thermal_authoring.yaml'}}
    stage.GetRootLayer().Save()
    for row in source['surfaces']:assert stage.GetPrimAtPath(row['usd_prim_path']).IsActive(),row['surface_id']
    for view,oldname in [('tank','model02_v4_tank_scene.yaml'),('detail','model02_v4_detail_scene.yaml')]:
        config=(ROOT/'configs'/oldname).read_text(encoding='utf-8')
        oldpath='../factory/model02_thermal_v4/model02_v4_scene.usda'
        assert config.count(oldpath)==1
        config=config.replace(oldpath,'../factory/model02_thermal_v5/model02_v5_scene.usda')
        path=ROOT/'configs'/('model02_v5_'+view+'_scene.yaml');assert not path.exists()
        path.write_text(config,encoding='utf-8')
    manifest={'result':'PASS_V5_GROUND_COMPOSITION','units':'m','up_axis':'Z','surface_count':len(source['surfaces']),
        'modified_paths':paths,'pbr_maps':sorted(set(maps)),'source_v4_stage_sha256':sha(OLD/'model02_v4_scene.usda'),
        'new_stage_sha256':sha(output),'source_v4_blend_sha256':sha(OLD/'model02_thermal_v4_configured.blend'),
        'thermal_configuration_sha256':sha(ROOT/'configs/model02_v4_thermal.yaml'),
        'note':'Material and shallow ground refinement only; retain v4 independent target IDs, temperatures, camera mount, start pose, lights and manual-preview workflow.'}
    (OUT/'ground_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('V5 COMPOSITION PASS',len(paths),'stable paths;',len(set(maps)),'portable PBR maps')


if __name__=='__main__':main()
