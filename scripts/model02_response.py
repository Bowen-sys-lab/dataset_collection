"""Require an independently bound visible region to be the sole temperature change."""
import numpy as np


def check_local_temperature_response(before,after,mask,delta_k):
    before=np.asarray(before);after=np.asarray(after);mask=np.asarray(mask,dtype=bool)
    if before.shape!=after.shape or before.shape!=mask.shape or not mask.any():raise ValueError('Missing or misaligned visible region')
    change=after-before
    if not np.isfinite(change).all() or not np.allclose(change[mask],delta_k,atol=.0001,rtol=0):raise ValueError('Selected region did not receive its temperature change')
    if np.any(np.abs(change[~mask])>.0001):raise ValueError('Other temperature pixels changed')
    return {'selected_gbuffer_pixels':int(mask.sum()),'changed_temperature_pixels':int((np.abs(change)>.0001).sum()),
            'maximum_other_temperature_change_k':float(np.max(np.abs(change[~mask]),initial=0))}
