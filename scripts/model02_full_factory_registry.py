"""Source completeness and identity checks for full-factory authoring."""
from collections import Counter


def validate_authoring_coverage(registry,batches,retained_sources):
    sources={r['source_object'] for r in registry['sources']};required={r['source_object'] for r in registry['sources'] if r.get('render_visible_original',True)}
    replaced=[name for b in batches for name in b['replaced_parts']]
    if len(replaced)!=len(set(replaced)):raise ValueError('duplicate source-part replacement')
    if required-set(replaced):raise ValueError('unprocessed raw sources: '+str(sorted(required-set(replaced))))
    parts={r['source_part'] for b in batches for r in b['surfaces']}
    if set(replaced)-parts:raise ValueError('replaced source part has no output')
    represented={r['source_object'] for b in batches for r in b['surfaces']}|set(retained_sources)
    if sources-represented:raise ValueError('source has no new or retained surfaces')
    return {'raw_sources':len(required),'source_parts':len(replaced),'sources_with_outputs':len(sources)}


def validate_registry(inventory,registry):
    sources={r['name']:r for r in inventory['objects']};rows=registry['sources']
    counts=Counter(r['source_object'] for r in rows)
    if any(n!=1 for n in counts.values()):raise ValueError('duplicate source registry entry')
    if set(counts)!=set(sources):raise ValueError('missing or unknown source registry entry')
    surface_ids=set();targets=set();pending=[]
    for r in rows:
        source=sources[r['source_object']];coverage=[0]*source['faces']
        for item in r['face_accounting']:
            a,b=item['start'],item['stop']
            if not 0<=a<b<=len(coverage):raise ValueError('invalid source face range')
            for i in range(a,b):coverage[i]+=1
        if any(n!=1 for n in coverage):raise ValueError('missing or duplicate source faces: '+r['source_object'])
        resolved=r['role']!='unclassified' and bool(r['classification_evidence'])
        if not resolved:
            pending.append(r['source_object'])
            if r['status'] in ['authored','bound','validated']:raise ValueError('unresolved classification claimed complete')
        if r['status'] in ['authored','bound','validated'] and not r.get('outputs'):raise ValueError('completed source has no actual output: '+r['source_object'])
        for item in r.get('outputs',[]):
            if item['surface_id'] in surface_ids:raise ValueError('duplicate surface identity')
            if item['target'] in targets:raise ValueError('duplicate thermal target identity')
            surface_ids.add(item['surface_id']);targets.add(item['target'])
    return {'source_objects':len(rows),'source_faces':sum(r['faces'] for r in sources.values()),
        'unclassified':pending,'complete':not pending and all(r['status']=='validated' for r in rows)}
