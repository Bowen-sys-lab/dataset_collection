"""Resolve append-only raw attempts into one explicitly corrected branch."""
import copy
import math


def active_branch(records, metadata):
    rows=[]
    offset=0.
    events=metadata.get('collection_edit_events',[])
    undos=[event for event in events if event['type']=='undo']
    next_event=0
    for raw in records:
        while next_event<len(undos) and raw['frame_id']>=undos[next_event]['next_raw_frame_id']:
            event=undos[next_event]
            index=next((i for i,row in enumerate(rows) if row['frame_id']==event['anchor_frame_id']),None)
            if index is None:raise ValueError('Undo anchor is not on the active branch')
            rows=rows[:index+1]
            offset=float(event['simulation_time_s'])-rows[-1]['simulation_time']
            next_event+=1
        row=copy.deepcopy(raw)
        if events:
            row['source_simulation_time_s']=raw['simulation_time']
            row['simulation_time']=float(raw['simulation_time'])-offset
        rows.append(row)
    # An undo may be the last event, with no new 1 Hz sample yet.
    while next_event<len(undos):
        event=undos[next_event]
        index=next((i for i,row in enumerate(rows) if row['frame_id']==event['anchor_frame_id']),None)
        if index is None:raise ValueError('Undo anchor is not on the active branch')
        rows=rows[:index+1]
        offset=float(event['simulation_time_s'])-rows[-1]['simulation_time']
        next_event+=1
    return rows,offset


def append_event(metadata, event):
    stamp=float(event['simulation_time_s'])
    events=metadata.setdefault('collection_edit_events',[])
    if not math.isfinite(stamp) or events and stamp<events[-1]['simulation_time_s']:
        raise ValueError('Edit event times must be finite and monotonic')
    event=dict(event,event_id=len(events))
    events.append(event)
    metadata['collection_edit_schema']='collision_undo_v1'
    return event
