"""A stopped-state inspection snapshot; no thermal render product while cruising."""
import time
from contextlib import contextmanager


def bind_inspection_history(recorder,events):
    recorder.metadata['preview_inspection_events']=events


@contextmanager
def frozen_initialization(orchestrator,reader_module,app):
    """Make vendor attachment warm-up zero-time, with partial-attach cleanup."""
    original_step=orchestrator.step
    original_reader=reader_module.AovReader
    counters={'explicit_render_calls':0}
    class SafeReader(original_reader):
        def attach(self,*args,**kwargs):
            try:return super().attach(*args,**kwargs)
            except BaseException:
                self.detach()
                raise
    def step(*args,**kwargs):
        for _ in range(3):app.update()
        kwargs.update(delta_time=0.,pause_timeline=True,wait_for_render=True)
        counters['explicit_render_calls']+=1
        return original_step(*args,**kwargs)
    orchestrator.step=step;reader_module.AovReader=SafeReader
    try:yield counters
    finally:
        orchestrator.step=original_step;reader_module.AovReader=original_reader


def release_thermal(thermal):
    try:
        thermal.close()
    finally:
        product=getattr(thermal,'_render_product',None)
        if product is not None:
            product.destroy()
            thermal._render_product=None


def capture_inspection(preview,timeline,app,controller):
    was_playing=timeline.is_playing()
    started=time.perf_counter()
    controller._suspend()  # Never resume with a held movement key or stale velocity.
    timeline.pause();timeline.commit();app.update()
    timestamp=float(timeline.get_current_time())
    event={'simulation_time_before':timestamp,'status':'FAILED','thermal_resources_released':False}
    try:
        with preview.inspection_initialization(app) as initialization:
            preview.thermal.open(settle_frames=4,rt_subframes=1)
        event['explicit_render_calls']=(initialization or {}).get('explicit_render_calls',0)
        # Drain authored camera/AOV changes without advancing simulation time.
        for _ in range(2):
            event['explicit_render_calls']+=1
            rgb,thermal,meta=preview.capture(render=True,simulation_time=timestamp,frozen_state=True,publish=False)
        if abs(float(timeline.get_current_time())-timestamp)>1e-8:
            raise RuntimeError('Inspection capture advanced simulation time')
        if not meta.get('thermal_coverage',{}).get('complete_for_this_view'):
            raise RuntimeError('Inspection has incomplete thermal coverage; confirm in offline replay')
        preview.publish_thermal(thermal,meta)
        event.update(status='COMPLETED',camera_world_matrix=meta.get('camera_world_matrix'),
            thermal_elapsed_s=meta.get('thermal_elapsed_s'),
            frozen_state_capture=meta.get('frozen_state_capture'),
            thermal_capture_drift_s=meta.get('simulation_capture_drift_s'),
            strict_sync_verified=False)
        return event
    finally:
        try:
            release_thermal(preview.thermal)
            event['thermal_resources_released']=True
            event['simulation_time_after']=float(timeline.get_current_time())
            event['wall_duration_s']=time.perf_counter()-started
            app.update()
        finally:
            if was_playing:timeline.play();timeline.commit()
