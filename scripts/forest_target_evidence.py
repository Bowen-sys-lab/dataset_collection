"""Selected surface identity/temperature evidence, separate from human task success."""
import numpy as np


def target_surface_evidence(instance_ids,labels,temperature_k,prim_path,expected_temperatures_c):
    ids=np.asarray(instance_ids);temp=np.asarray(temperature_k,dtype=float)
    if ids.shape!=temp.shape or ids.ndim!=2 or not np.isfinite(temp).all():
        raise ValueError('Finite equal-size target ID and truth-temperature planes are required')
    if not isinstance(prim_path,str) or not prim_path.startswith('/'):
        raise ValueError('An absolute selected target Prim path is required')
    expected=np.asarray(expected_temperatures_c,dtype=float)+273.15
    if expected.ndim!=1 or not expected.size or not np.isfinite(expected).all():
        raise ValueError('Finite prescribed target temperatures are required')
    selected=[int(k) for k,v in labels.items() if str(v)==prim_path or str(v).startswith(prim_path+'/')]
    mask=np.isin(ids,selected)
    matches=np.zeros(temp.shape,dtype=bool)
    for value in expected:matches|=np.isclose(temp,value,atol=.2,rtol=0)
    pixels=int(mask.sum());temperature_pixels=int((mask&matches).sum())
    return dict(result='PASS' if pixels and temperature_pixels else 'FAIL',
                target_prim_path=prim_path,target_pixels=pixels,
                expected_temperature_pixels=temperature_pixels,
                scope='Selected visible rendered surfaces and prescribed truth; not automatic task success or injury diagnosis')
