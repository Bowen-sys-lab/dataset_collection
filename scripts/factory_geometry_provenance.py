"""Pure file binding for a static composed factory geometry cache and its route audit."""
import hashlib
from pathlib import Path


def fingerprint_inputs(paths,root):
    root=Path(root).resolve();result={}
    for value in paths:
        path=Path(value).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Geometry source is missing or outside project: '+str(path))
        result[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not result:raise ValueError('Composed geometry fingerprint is empty')
    return result


def cache_sources_current(metadata,geometry_file,root):
    try:
        root=Path(root).resolve();geometry_file=Path(geometry_file).resolve()
        sources=metadata['geometry_source_inputs'];stage=metadata['stage_relative_path']
        code=metadata['geometry_source_code_inputs']
        if metadata.get('geometry_source_schema_version')!=1 or not isinstance(sources,dict) or not sources:return False
        if not isinstance(code,dict) or not code or fingerprint_inputs([root/key for key in code],root)!=code:return False
        if stage not in sources or sources[stage]!=metadata['stage_sha256']:return False
        if not geometry_file.is_relative_to(root) or not geometry_file.is_file():return False
        current=fingerprint_inputs([root/key for key in sources],root)
        if current!=sources:return False
        return hashlib.sha256(geometry_file.read_bytes()).hexdigest()==metadata['geometry_cache_sha256']
    except (OSError,ValueError,TypeError,KeyError):return False


def route_audit_current(audit,metadata,geometry_file,root):
    try:
        root=Path(root).resolve();code=audit['route_source_code_inputs']
        return bool(isinstance(code,dict) and code and fingerprint_inputs([root/key for key in code],root)==code
            and cache_sources_current(metadata,geometry_file,root)
            and audit.get('stage_sha256')==metadata.get('stage_sha256')
            and audit.get('geometry_sha256')==metadata.get('geometry_cache_sha256')
            and audit.get('geometry_source_inputs')==metadata.get('geometry_source_inputs'))
    except (OSError,ValueError,TypeError,KeyError):return False
