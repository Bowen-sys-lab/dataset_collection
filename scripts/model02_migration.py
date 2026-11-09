"""Strict stable-ID binding to paths read from the exported asset."""
import hashlib
from datetime import datetime,timedelta,timezone


def night_weather(condition):
    hours=condition['hours_after_sunset']
    if not 0<=hours<=10:raise ValueError('Nominal night fixture supports0–10hours after assumed18:00 sunset; other conditions need review')
    start=datetime(2026,10,18,10,tzinfo=timezone.utc)+timedelta(hours=hours)
    text='# irsim weather v1\n# Synthetic dry operating night; assumed sunset18:00 UTC+8. Not field weather.\n'
    text+='time_utc,t_air_c,rh_fraction,wind_speed_m_s,cloud_fraction,dni_w_m2,dhi_w_m2,visibility_m,precip_mm_h\n'
    air=float(condition['air_temperature_c']);wind=float(condition['wind_m_s'])
    for offset in [-2,0,4]:
        stamp=(start+timedelta(hours=offset)).strftime('%Y-%m-%dT%H:%M:%SZ')
        text+=f'{stamp},{air},0.60,{wind},0.10,0,0,23000,0\n'
    return text,start.strftime('%Y-%m-%dT%H:%M:%SZ')


def validate_export_snapshot(exported,blend_file,asset_file):
    # Temperatures/material physics can change without re-exporting RGB geometry.
    for path,key in [(blend_file,'configured_blend_sha256'),(asset_file,'asset_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=exported[key]:raise ValueError('Exported geometry snapshot changed: '+str(path))


def core_temperature_truth(entry,authoring):
    from scripts.model02_authoring import resolve_temperature
    values={sid:resolve_temperature(authoring['targets'][authoring['surfaces'][sid]['target']],
                authoring['condition']['air_temperature_c']) for sid in entry['target_surface_ids']}
    low,high=min(values.values()),max(values.values())
    return {'hotspot_temperature_c':low if low==high else None,
            'hotspot_temperature_range_c':[low,high],'hotspot_temperatures_c':values}


def bind_surface_rows(rows,authoring):
    surfaces=authoring['surfaces'];by_id={r['surface_id']:r for r in rows}
    if len(by_id)!=len(rows) or set(by_id)!=set(surfaces):raise ValueError('Missing, duplicate or unknown exported surface IDs')
    bindings=[]
    for sid,item in surfaces.items():
        row=by_id[sid]
        if row['object']!=item['object']:raise ValueError('Stale exported object identity: '+sid)
        bindings.append({'name':item['target'],'prim_path':row['usd_prim_path'],
                         'include_children':False,'target':item['target'],'material':item['material']})
    if len({b['prim_path'] for b in bindings})!=len(bindings):raise ValueError('Repeated exported mesh path')
    return bindings
