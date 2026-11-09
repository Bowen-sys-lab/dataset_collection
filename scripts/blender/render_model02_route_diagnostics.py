"""Actual Blender RGB construction views; no thermal rendering claim."""
import bpy,json
from pathlib import Path
from mathutils import Vector
OUT=Path('D:/IsaacSim_Demo/factory/model02_thermal_v3')
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=24;s.cycles.use_denoising=True
s.render.resolution_x=1200;s.render.resolution_y=900;s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG'
cam=s.camera;cam.data.sensor_width=36.;cam.data.clip_end=10000

def capture(name,eye,target,lens):
    cam.location=eye;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.lens=lens
    bpy.context.view_layer.update();s.render.filepath=str(OUT/name)
    bpy.ops.render.render(write_still=True)
    print('Saved '+name,flush=True)

capture('weaklight_overview.png',(646,-867,44),(700,-809,3),32)
capture('tank_observation_rgb.png',(693.37,-797.25,3.5),(693.674,-789.905,2.834),32)
capture('pipe_observation_rgb.png',(688,-804.3,3.5),(682.28082,-799.35732,3.5),32)
s.world=s.world.copy();background=next(n for n in s.world.node_tree.nodes if n.type=='BACKGROUND')
background.inputs['Color'].default_value=(.8,.85,1.,1.);background.inputs['Strength'].default_value=.5
capture('pipe_construction_detail.png',(684.1,-802.7,3.8),(682.28082,-799.35732,3.5),70)
capture('facade_construction_detail.png',(727,-843,6.5),(717,-829,3),48)
(OUT/'render_notes.json').write_text(json.dumps({'images':{'weaklight_overview.png':'Actual Blender RGB, saved scene night world and13 authored lamps','tank_observation_rgb.png':'Actual Blender RGB from the saved tank task observation point','pipe_observation_rgb.png':'Actual Blender RGB from the saved pipe task observation point','pipe_construction_detail.png':'Actual Blender RGB, temporary neutral inspection illumination to see jacket/wool/existing pipe; not saved back to scene','facade_construction_detail.png':'Actual Blender RGB with temporary inspection illumination'},'thermal_images':False,'saved_blend_lighting_changed':False},indent=2),encoding='utf-8')
