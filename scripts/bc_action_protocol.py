"""Versioned signed pose targets for the future single-front BC model."""
import math
from numbers import Real
import re
import numpy as np

SCHEMA='factory_pose_action_v1'
SCHEMA_V2='factory_pose_action_v2'
AXES=('forward_m','left_m','up_m','yaw_delta_rad','duration_s')
MAX_HORIZONTAL_SPEED_M_S=3.1
MAX_VERTICAL_SPEED_M_S=1.05
MAX_YAW_SPEED_RAD_S=.8
MAX_DURATION_S=.75
MAX_DURATION_BY_SCHEMA={SCHEMA:MAX_DURATION_S, SCHEMA_V2:1.25}
_DECIMAL=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
_POSE=re.compile(r'POSE\s+'+r'\s+'.join([f'({_DECIMAL})']*5))


def _number(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not math.isfinite(value):
        raise ValueError('Action values must be finite real numbers')
    return float(value)


def checked_action(action, *, schema=SCHEMA):
    if not isinstance(schema,str) or schema not in MAX_DURATION_BY_SCHEMA:
        raise ValueError('Unknown action schema')
    if type(action.get('stop')) is not bool:
        raise ValueError('Explicit boolean stop field required')
    values={key:_number(action[key]) for key in AXES}
    fwd,left,up,yaw,duration=(values[key] for key in AXES)
    if action['stop']:
        if any(value!=0. for value in values.values()):
            raise ValueError('STOP cannot contain movement or duration')
        kind='STOP'
    else:
        if not 0 < duration <= MAX_DURATION_BY_SCHEMA[schema]:
            raise ValueError(f'POSE duration must be in (0,{MAX_DURATION_BY_SCHEMA[schema]}] seconds for {schema}')
        if (math.hypot(fwd,left)/duration > MAX_HORIZONTAL_SPEED_M_S+1e-6
            or abs(up)/duration > MAX_VERTICAL_SPEED_M_S+1e-6
            or abs(yaw)/duration > MAX_YAW_SPEED_RAD_S+1e-6):
            raise ValueError('POSE exceeds the declared diagnostic rate limits')
        kind='HOLD' if math.sqrt(fwd*fwd+left*left+up*up)<=1e-4 and abs(yaw)<=1e-4 else 'MOVE'
    return {**values,'stop':action['stop'],'motion_kind':kind}


def encode_action(action, *, schema=SCHEMA):
    values=checked_action(action,schema=schema)
    if values['stop']:
        return 'STOP'
    text='POSE '+' '.join(f'{values[key]:+.9f}' for key in AXES)
    # Reject a rounding-induced invalid duration/rate; never silently clip it.
    parse_action(text,schema=schema)
    return text


def parse_action(text, *, schema=SCHEMA):
    if not isinstance(text,str) or len(text)>256:
        raise ValueError('Bounded action text required')
    text=text.strip()
    if text.endswith('</s>'):
        text=text[:-4].strip()
    if text=='STOP':
        return checked_action({**dict.fromkeys(AXES,0.),'stop':True},schema=schema)
    match=_POSE.fullmatch(text)
    if match is None:
        raise ValueError('Expected POSE with5 decimal numbers or STOP')
    return checked_action({**dict(zip(AXES,map(float,match.groups()))),'stop':False},schema=schema)


def plan_action(position,yaw,action,*,step_s=1/60,schema=SCHEMA):
    """Simultaneous straight world translation/yaw; collision guards remain external."""
    action=checked_action(action,schema=schema)
    position=np.asarray(position,dtype=float)
    yaw,step_s=_number(yaw),_number(step_s)
    if position.shape!=(3,) or not np.isfinite(position).all() or step_s<=0:
        raise ValueError('Finite3D position and positive sampling step required')
    poses=[{'time_s':0.,'position':position.tolist(),'yaw':yaw}]
    if action['stop']:
        return {'poses':poses,'phases':[],'stop_requested':True,'translation_executed':False,'action_schema':schema}
    duration=action['duration_s']
    ratio=duration/step_s
    if not math.isfinite(ratio) or ratio>9999:
        raise ValueError('Motion exceeds10000-pose budget')
    count=max(1,math.ceil(ratio))
    c,s=math.cos(yaw),math.sin(yaw)
    offset=np.array([c*action['forward_m']-s*action['left_m'],
                     s*action['forward_m']+c*action['left_m'],action['up_m']])
    with np.errstate(over='raise',invalid='raise'):
        target=position+offset
    if not np.isfinite(target).all() or not np.allclose(target-position,offset,rtol=0,atol=1e-6):
        raise ValueError('Position precision cannot represent requested motion')
    target_yaw=yaw+action['yaw_delta_rad']
    if not math.isfinite(target_yaw) or (target_yaw==yaw and action['yaw_delta_rad']!=0):
        raise ValueError('Heading precision cannot represent requested turn')
    for index in range(1,count+1):
        fraction=index/count
        poses.append({'time_s':duration*fraction,'position':(position+fraction*offset).tolist(),
                      'yaw':yaw+fraction*action['yaw_delta_rad']})
    return {'poses':poses,'phases':[{'name':'pose_target','duration_s':duration,'first_pose':0,'last_pose':count}],
            'stop_requested':False,'translation_executed':bool(np.any(offset!=0)), 'action_schema':schema,
            'coordinate_convention':'yaw-local +forward/+left, world +up, +yaw counter-clockwise',
            'scope':'Kinematic endpoint plan; straight segments must pass continuous collision checks'}
