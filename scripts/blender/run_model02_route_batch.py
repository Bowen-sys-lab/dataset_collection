"""Native Blender background continuation when the interactive MCP is waiting."""
import bpy
import sys
import time
import json
import traceback
from pathlib import Path

sys.path.insert(0,'D:/IsaacSim_Demo')
from scripts.blender import model02_route_refinement as v3
ROOT=Path('D:/IsaacSim_Demo/factory/model02_thermal_v3_build')
ROOT.mkdir(exist_ok=True)
v3.ROOT=ROOT
try:
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'model02_thermal_v3.blend'))
    print('V3 begin',flush=True)
    print(v3.setup(),flush=True)
    print('V3 refine existing',flush=True)
    print(v3.refine_existing(),flush=True)
    for name in ['11_29_gc089','11_29_gc090','11_29_gc092','11_29_gc093','11_29_gc010']:
        print('V3 source '+name,flush=True)
        print(v3.refine_source(name),flush=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'model02_thermal_v3.blend'))
    print('V3 pipe failure',flush=True)
    pipe=v3.add_pipe_failure()
    (ROOT/'pipe_construction.json').write_text(json.dumps(pipe,indent=2),encoding='utf-8')
    print(pipe,flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'model02_thermal_v3.blend'))
    (ROOT/'batch_result.json').write_text(json.dumps({'result':'GEOMETRY_BUILT_PENDING_VALIDATION'}),encoding='utf-8')
    print('V3 geometry saved',flush=True)
except BaseException as exc:
    (ROOT/'batch_result.json').write_text(json.dumps({'result':'FAIL','error':str(exc)}),encoding='utf-8')
    traceback.print_exc()
    raise
