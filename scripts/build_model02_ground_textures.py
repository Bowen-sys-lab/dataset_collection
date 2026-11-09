"""Portable, tileable PBR maps with restrained colour and soft surface relief."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'factory/model02_thermal_v5'


def main():
    folder=OUT/'textures';folder.mkdir(parents=True,exist_ok=True)
    n=1024;x,y=np.meshgrid(np.arange(n)/n,(n-1-np.arange(n))/n)
    specifications={'brick':([151,148,139],.86,.00012,2.432),
        'concrete':([137,135,128],.9,.00018,4.01),
        'joint':([111,108,100],.95,.00006,2.432),
        'grass':([72,91,47],.97,.003,2.432)}
    report={}
    for kind,(color,rough,amplitude,meters) in specifications.items():
        rng=np.random.default_rng(870+list(specifications).index(kind))
        h=np.zeros_like(x);dx=np.zeros_like(x);dy=np.zeros_like(x)
        for _ in range(12):
            a,b=rng.integers(2,18,size=2);phase=rng.uniform(0,2*np.pi)
            if kind=='grass' and _>5:a,b=rng.integers(25,65),rng.integers(5,14)
            angle=2*np.pi*(a*x+b*y)+phase;scale=amplitude/12
            h+=scale*np.sin(angle)
            dx+=scale*np.cos(angle)*2*np.pi*a/meters
            dy+=scale*np.cos(angle)*2*np.pi*b/meters
        normal=np.stack([-dx,-dy,np.ones_like(x)],axis=-1)
        normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
        variation=(h/max(amplitude,1e-10))*(.06 if kind=='grass' else .018)
        rgb=np.clip(np.array(color)[None,None,:]*(1+variation[:,:,None]),0,255).astype('uint8')
        Image.fromarray(rgb).save(folder/(kind+'_basecolor.png'))
        Image.fromarray(np.rint((normal+1)*127.5).astype('uint8')).save(folder/(kind+'_normal.png'))
        Image.fromarray(np.uint8(np.clip(rough+variation*.1,0,1)*255)).save(folder/(kind+'_roughness.png'))
        report[kind]={'texture_span_m':meters,'roughness':rough,'surface_relief_amplitude_m':amplitude,
            'interpretation':'synthetic soft canopy shading' if kind=='grass' else 'restrained soft surface waviness; no emphasized grain'}
    (OUT/'texture_parameters.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PBR maps generated:',len(report)*3)


if __name__=='__main__':main()
