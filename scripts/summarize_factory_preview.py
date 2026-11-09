"""Compare the six native paired/split measurements against the approved gates."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(folder):
    rows=[]; condition=None
    for i in range(1,4):
        pair={}
        for mode in ('paired','split'):
            r=json.loads((folder/f'{mode}-{i}/performance.json').read_text(encoding='utf-8'))
            if r['completion']!='completed' or r.get('benchmark_measurement_s',0)<60:
                raise ValueError('Six completed post-warmup measurements >=60 seconds are required')
            signature=(r['last_frame_metadata']['scene_config_sha256'],r['renderer'],tuple(r['resolution']),
                       json.dumps(r['controller_parameters'],sort_keys=True))
            if condition is None:condition=signature
            if condition!=signature:raise ValueError('Benchmark conditions differ')
            c=r['control_metrics']['controller_wall_step_interval_s']
            pair[mode]={'control_mean_s':c['mean'],'control_p95_s':c['p95'],
                'rgb_refresh_fps':r.get('rgb_refresh_fps'),'thermal_refresh_fps':r.get('thermal_refresh_fps'),
                'measurement_s':r['benchmark_measurement_s'],
                'thermal_get_outputs_mean_s':r['stage_timings_s']['thermal_get_outputs_s']['mean']}
        pair['round']=i
        pair['mean_improvement_fraction']=1-pair['split']['control_mean_s']/pair['paired']['control_mean_s']
        pair['p95_ratio']=pair['split']['control_p95_s']/pair['paired']['control_p95_s']
        rows.append(pair)
    baseline=statistics.mean(r['paired']['control_mean_s'] for r in rows)
    split=statistics.mean(r['split']['control_mean_s'] for r in rows)
    improvement=1-split/baseline
    passed=improvement>=.2 and all(r['p95_ratio']<=1.1 for r in rows)
    return {'result':'PASS' if passed else 'FAIL','mean_improvement_fraction':improvement,
            'paired_mean_control_interval_s':baseline,'split_mean_control_interval_s':split,
            'maximum_p95_ratio':max(r['p95_ratio'] for r in rows),'rounds':rows,
            'scope':'headless native app/control timings with synthetic wall-clock inputs; not human end-to-end latency',
            'source_frame_freshness_verified':False,'manual_demonstrations_created':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    args=parser.parse_args();report=summarize(args.folder)
    (args.folder/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
