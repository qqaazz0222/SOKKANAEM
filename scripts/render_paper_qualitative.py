"""Render verified qualitative replay tensors; does not perform inference."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paper_qualitative import OUT, DEST, SUP, MODELS, NAMES, write_json
from scripts.paper_closeout_study import verify_files
from sokkanaem.protocol import file_record


def render():
    sys.path.insert(0, str(Path('work_dirs/paper_tools/pdf_python').resolve()))
    import fitz
    import cv2
    audit = json.loads((OUT/'inference.json').read_text())
    verify_files([audit['generator'], audit['selection'], *audit['inputs'], *audit['artifacts']])
    DEST.mkdir(parents=True, exist_ok=True)
    SUP.mkdir(parents=True, exist_ok=True)
    ink, muted = (.133,.208,.302), (.322,.392,.467)
    def color_depth(a, valid=None):
        values = np.clip(a.squeeze()/5, 0, 1)
        color = cv2.applyColorMap(np.rint(values*255).astype('uint8'), cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
        if valid is not None:
            color[~valid.squeeze()] = [210,215,220]
        return color
    def rgb_image(a):
        return np.rint(np.clip(a.transpose(1,2,0),0,1)*255).astype('uint8')
    def mask_image(a):
        m = cv2.resize(a.reshape(16,16).astype('uint8'), (256,256), interpolation=cv2.INTER_NEAREST)
        return np.where(m[...,None] > 0, np.array([8,127,140], dtype='uint8'), np.array([231,237,242], dtype='uint8'))
    def put(page, x, y, a, size=115):
        h,w = a.shape[:2]
        pix = fitz.Pixmap(fitz.csRGB,w,h,a.tobytes(),False)
        page.insert_image(fitz.Rect(x,y,x+size,y+size), pixmap=pix)
    def label(page,x,y,s,size=10,bold=False):
        page.insert_text((x,y),s,fontsize=size,fontname='hebo' if bold else 'helv',color=ink if bold else muted)
    def legend(page,y,width):
        ramp = np.repeat(np.arange(256,dtype='uint8')[None,:],12,axis=0)
        color = cv2.applyColorMap(ramp,cv2.COLORMAP_VIRIDIS)[...,::-1].copy()
        pix = fitz.Pixmap(fitz.csRGB,256,12,color.tobytes(),False)
        page.insert_image(fitz.Rect(120,y,376,y+12),pixmap=pix)
        label(page,120,y+25,'0 m',8); label(page,354,y+25,'5 m',8)
        label(page,405,y+10,'Fixed clip-aligned depth range; values >5 m saturated',9)
        label(page,405,y+24,'Gray GT: invalid pixels. Not calibration-free metric depth.',8)
    arrays = {}
    for rec in audit['clips']:
        stem = rec['id'].replace(':','_')
        data = dict(np.load(OUT/(stem+'_input.npz')))
        for name in MODELS:
            data[name] = dict(np.load(OUT/(stem+'_'+name+'.npz')))
        arrays[rec['id']] = data
    def images(data,t):
        return [rgb_image(data['rgb'][t]),color_depth(data['gt'][t],data['valid'][t])]+[
            color_depth(data[m]['depth'][t]) for m in MODELS]
    artifacts = []
    doc = fitz.open(); page = doc.new_page(width=830,height=590)
    for j,n in enumerate(('RGB','GT',*NAMES)):
        label(page,105+j*120,22,n,10,True)
    for i,rec in enumerate(audit['clips']):
        y=35+i*125
        label(page,8,y+43,rec['sequence'].replace('rgbd_dataset_freiburg3_','').replace('_',' '),8)
        label(page,8,y+60,rec['id']+' / t=127',8)
        for j,img in enumerate(images(arrays[rec['id']],127)):
            put(page,105+j*120,y,img)
    legend(page,548,830)
    path=DEST/'qualitative_spatial.pdf'; doc.save(path,no_new_id=True,deflate=True)
    doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(DEST/'qualitative_spatial.png');doc.close()
    artifacts.extend(file_record(DEST/('qualitative_spatial'+s)) for s in ('.pdf','.png'))
    doc=fitz.open();page=doc.new_page(width=730,height=810)
    data=arrays['tum:7']; times=(28,29,30,31,32)
    for j,t in enumerate(times):
        label(page,115+j*122,20,f't={t}'+(' (keyframe)' if t==30 else ''),10,True)
    for i,n in enumerate(('RGB','GT',*NAMES,'Active mask')):
        y=32+i*103
        label(page,8,y+40,n,10,True)
        for j,t in enumerate(times):
            img=images(data,t)[i] if i<6 else mask_image(data['sparse_k30']['mask'][t])
            put(page,115+j*122,y,img,size=98)
            if i==6:
                label(page,115+j*122,y+108,f"{data['sparse_k30']['mask'][t].mean()*100:.1f}% active",8)
    legend(page,775,730)
    path=DEST/'qualitative_temporal.pdf';doc.save(path,no_new_id=True,deflate=True)
    doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(DEST/'qualitative_temporal.png');doc.close()
    artifacts.extend(file_record(DEST/('qualitative_temporal'+s)) for s in ('.pdf','.png'))
    for rec in audit['clips']:
        data=arrays[rec['id']]; stem=rec['id'].replace(':','_')
        path=SUP/(stem+'.mp4')
        cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x570',
             '-framerate','10','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(path)]
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        for t in range(256):
            canvas=np.full((570,1024,3),255,dtype='uint8')
            panels=images(data,t)+[mask_image(data['sparse_k30']['mask'][t])]
            names=['RGB','GT',*NAMES,'Active mask']
            for k,(a,n) in enumerate(zip(panels,names)):
                x=(k%4)*256;y=(k//4)*282
                cv2.putText(canvas,n,(x+8,y+18),cv2.FONT_HERSHEY_SIMPLEX,.48,(34,53,77),1,cv2.LINE_AA)
                canvas[y+24:y+280,x:x+256]=a
            cv2.putText(canvas,rec['id']+f'  t={t}  / 255',(778,312),cv2.FONT_HERSHEY_SIMPLEX,.45,(34,53,77),1)
            cv2.putText(canvas,'0-5 m fixed; GT fit / clip',(778,343),cv2.FONT_HERSHEY_SIMPLEX,.40,(34,53,77),1)
            cv2.putText(canvas,'10 fps presentation',(778,374),cv2.FONT_HERSHEY_SIMPLEX,.45,(34,53,77),1)
            cv2.putText(canvas,'Not inference speed',(778,405),cv2.FONT_HERSHEY_SIMPLEX,.45,(34,53,77),1)
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        if process.wait()!=0: raise RuntimeError('video encoding failed')
        artifacts.append(file_record(path))
        print('Rendered video',rec['id'],flush=True)
    # Portable compact provenance excludes multi-GB replay tensors, retains IDs/hashes/fit/clipping.
    write_json(SUP/'provenance.json',dict(protocol=file_record('paper/QUALITATIVE_PLAN.md'),
        inference=file_record(OUT/'inference.json'),clips=audit['clips'],post_hoc=True,
        depth_range_m=[0,5],per_frame_alignment=False,playback_fps=10))
    artifacts.append(file_record(SUP/'provenance.json'))
    write_json(OUT/'render.json',dict(generator=file_record(__file__),inputs=[file_record(OUT/'inference.json'),
        *audit['artifacts']],artifacts=artifacts,post_hoc=True,synthetic_predictions=False))
    print('Rendered two figures and four videos',flush=True)


if __name__ == '__main__':
    render()
