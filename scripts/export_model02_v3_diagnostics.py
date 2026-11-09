"""Derive labelled narrow-window scientific displays from saved raw apparent-T arrays."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.thermal_display import fixed_temperature_display

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/isaac-thermal-camera-simulation/src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();folder=args.input
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    assert summary['result']=='PASS_ISAAC_MIGRATION_BINDING_AND_LOCAL_RESPONSE'
    manifest=json.loads((ROOT/'factory/model02_thermal_v3/isaac_manifest.json').read_text(encoding='utf-8'))
    rows={r['usd_prim_path']:r for r in manifest['surfaces']};report={'display_only':True,'range_c':[10,35],'components':[]}
    for index in [1,2]:
        stem=f'frame_{index:06d}'
        apparent=np.load(folder/(stem+'_apparent_t.npy'));surface=np.load(folder/(stem+'_surface_temperature_k.npy'))
        ids=np.load(folder/(stem+'_instance_id.npy'))
        labels=json.loads((folder/(stem+'_instance_paths.json')).read_text(encoding='utf-8'))
        assert ids.shape==apparent.shape
        output=folder/(stem+'_thermal_detail_10_35C.png')
        Image.fromarray(fixed_temperature_display(apparent,10.,35.,palette='gray',polarity='white_hot')).save(output)
        for component in ['Wall','WindowGlass','WindowFrame','DoorPanel','DoorFrame']:
            selected=[int(i) for i,p in labels.items() if p in rows and rows[p]['source_object']=='11_29_gc021' and rows[p]['component']==component]
            mask=np.isin(ids,selected)
            if mask.any():report['components'].append({'frame':index,'component':component,'pixels':int(mask.sum()),
                'surface_temperature_median_c':float(np.median(surface[mask])-273.15),
                'apparent_temperature_median_c':float(np.median(apparent[mask])-273.15)})
    (folder/'facade_detail_display.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    examples={r['kind']:{'target':r['target'],'surface_id':r['surface_id'],'usd_prim_path':r['usd_prim_path'],
        'baseline_temperature_c':r['original_temperature_c'],'example_test_temperature_c':r['test_temperature_c']}
        for r in summary['local_responses']}
    (ROOT/'factory/model02_thermal_v3/configuration/independent_edit_examples.json').write_text(json.dumps({
        'instruction':'Edit the matching target temperature_c in thermal_authoring.yaml, recompile then rebuild v3 configs; examples are not applied to baseline.',
        'examples':examples},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
