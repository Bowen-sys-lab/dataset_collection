"""Optional project material additions, preserving all default definitions."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'third_party/isaac-thermal-camera-simulation'


def load_project_material_library(directory=None):
    sys.path.insert(0,str(VENDOR/'src'))
    from irsim.materials.library import MaterialLibrary
    base=MaterialLibrary.load(data_dir=VENDOR/'data')
    if directory is None:return base
    custom=MaterialLibrary.load(directory,VENDOR/'data')
    for name in set(base)&set(custom):
        if base[name].content_hash()!=custom[name].content_hash():raise ValueError('Inconsistent material override: '+name)
    return MaterialLibrary({**dict(base),**dict(custom)})
