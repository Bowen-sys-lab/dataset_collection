"""Pure standard-library freshness guard, also usable in Blender's Python."""
import hashlib


def verify_inputs(root,data):
    inputs=data['input_sha256']
    for directory in data['material_directories']:
        current={p.relative_to(root).as_posix() for p in (root/directory).glob('*.yaml')}
        expected={p for p in inputs if p.startswith(directory+'/') and p.endswith('.yaml')}
        if current!=expected:raise ValueError('Material directory membership changed; recompile configuration')
    for name,expected in inputs.items():
        path=root/name
        if not path.is_file():raise ValueError('Configuration input missing: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Configuration input changed; recompile: '+name)
