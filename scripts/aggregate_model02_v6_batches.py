"""Require every exported part before publishing a complete zone manifest."""
import collections,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'factory/model02_thermal_v6'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def merge_reports(batch,reports):
    if not reports or any(r['result']!='PASS_NATIVE_V6_CHUNK_EXPORT' or r['batch_label']!=batch['label'] for r in reports):raise ValueError('Missing or wrong export part')
    total=reports[0]['parts_in_batch']
    if len(reports)!=total or {r['part_index'] for r in reports}!=set(range(total)) or any(r['parts_in_batch']!=total for r in reports):raise ValueError('Missing or duplicate export part')
    reports=sorted(reports,key=lambda r:r['part_index']);rows=[s for r in reports for s in r['surfaces']]
    if len({r['surface_id'] for r in rows})!=len(rows) or {r['surface_id'] for r in rows}!={r['surface_id'] for r in batch['surfaces']}:raise ValueError('Incomplete or duplicate exported surfaces')
    if len({r['usd_prim_path'] for r in rows})!=len(rows):raise ValueError('Exported path collision')
    audits={};counts=collections.Counter()
    for report in reports:
        for item in report['source_face_audit']:
            name=item['source_part'];counts[name]+=item['chunk_surfaces'];core={k:v for k,v in item.items() if k!='chunk_surfaces'}
            if name in audits and audits[name]!=core:raise ValueError('Conflicting source face evidence')
            audits[name]=core
    if counts!=collections.Counter(r['source_part'] for r in batch['surfaces']):raise ValueError('Source part audit coverage mismatch')
    return {'result':'PASS_NATIVE_V6_BATCH_EXPORT','label':batch['label'],'surfaces':rows,'replaced_parts':batch['replaced_parts'],'part_count':total,
        'configured_assets':[{'path':r['configured_asset'],'sha256':r['configured_asset_sha256'],'collection':'V6Asset_'+r['label']} for r in reports],
        'usd_files':[{'path':r['usd_file'],'sha256':r['usd_sha256']} for r in reports],
        'texture_dependencies':{r['label']:r['texture_dependencies'] for r in reports},'source_face_audit':[dict(audits[name],chunk_surfaces=counts[name]) for name in sorted(audits)]}


def main():
    jobs=json.loads((OUT/'configuration/export_jobs.json').read_text());completed=0
    for path in sorted((OUT/'batches').glob('*.json')):
        batch=json.loads(path.read_text());selected=[j for j in jobs if j['batch_label']==batch['label']];reports=[]
        for job in selected:
            report=json.loads((OUT/'usd'/job['label']/'export_manifest.json').read_text())
            assert sha(OUT/report['configured_asset'])==report['configured_asset_sha256'] and sha(OUT/report['usd_file'])==report['usd_sha256']
            reports.append(report)
        result=merge_reports(batch,reports);folder=OUT/'usd'/batch['label'];folder.mkdir(exist_ok=True)
        (folder/'export_manifest.json').write_text(json.dumps(result,indent=2));completed+=len(result['surfaces'])
    print('V6 verified complete batch exports',completed,flush=True)


if __name__=='__main__':main()
