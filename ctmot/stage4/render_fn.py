"""Render verified FN examples with the existing RGB projection/drawing helpers."""
import cv2
import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from .common import *
from utils.viz import project_box_edges, draw_segment, draw_text
from data.script.NUSC_CONSTANT import USED_ALLCAM
from geometry.nusc_utils import fast_project_dets_to_images
from pre_processing.data_fusion import datafusion

WHITE=(245,245,245);GREEN=(40,230,40);RED=(40,40,240);CYAN=(255,230,40);ORANGE=(30,150,255);PURPLE=(255,70,220)


def edges(obj,meta,w,h):
    box=Box(obj['translation'],obj['size'],Quaternion(obj['rotation']))
    return project_box_edges(box,meta['pose'],meta['calibration'],w,h,False)


def row_box(row):
    return dict(translation=row[:3],size=row[3:6],rotation=row[8:12])


def paint(img,obj,meta,color,dashed=False):
    es=edges(obj,meta,img.shape[1],img.shape[0])
    for a,b in es:draw_segment(img,a,b,color,dashed)
    return es


def roi_for(es,w,h):
    pts=np.asarray(es).reshape(-1,2);lo=pts.min(axis=0);hi=pts.max(axis=0);center=(lo+hi)/2
    cw=min(w,max(320.,(hi[0]-lo[0])*3.8,(hi[1]-lo[1])*3.8*1.75));ch=cw/1.75
    if ch>h:ch=h;cw=ch*1.75
    x=int(np.clip(center[0]-cw/2,0,w-cw));y=int(np.clip(center[1]-ch/2,0,h-ch))
    return [x,y,int(x+cw),int(y+ch)]


def panel(img,roi,lines):
    x,y,x2,y2=roi;crop=img[y:y2,x:x2]
    view=cv2.resize(crop,(800,457),interpolation=cv2.INTER_CUBIC)
    header=np.full((100,800,3),22,dtype=np.uint8)
    for i,line in enumerate(lines):draw_text(header,line,(12,23+23*i))
    return np.vstack([header,view])


def render_key(cfg,out,scene,frame,archive):
    group=scene['groups'][0];r=next(f for f in group['frames'] if f['frame_idx']==frame)
    gt=r['actual_gt'];assert gt is not None
    cameras=archive['cameras'][r['sample_token']];areas={}
    for cam,meta in cameras.items():
        es=edges(gt,meta,1600,900)
        if es:
            p=np.asarray(es).reshape(-1,2);areas[cam]=np.prod(p.max(axis=0)-p.min(axis=0))
    camera=max(areas,key=areas.get);meta=cameras[camera]
    src=resolve(cfg['dataset_root'])/meta['filename'];original=cv2.imread(str(src));assert original is not None
    roi=roi_for(edges(gt,meta,original.shape[1],original.shape[0]),original.shape[1],original.shape[0])
    parts=[];manifest=[]
    for v in ('B0','B1'):
        img=original.copy();t=r[v];tid=t['track_id'];threshold=next(s for s in read_json(out/'selection.json') if s['scene']==scene['scene'])['classes'][group['cls']]['threshold']
        predicted=t['prediction'];candidate=t['candidate'];raw=t['raw_output']
        paint(img,gt,meta,WHITE,True)
        if candidate and candidate['detection_gt_xy_m']<2.:
            paint(img,row_box(candidate['detection']),meta,GREEN,True)
        if predicted:paint(img,row_box(predicted),meta,RED,True)
        mean=next((b.tracking_score for ts in archive['predictions'][v][scene['scene_token']].values() for b in ts if b.tracking_id==tid),None)
        limit=50. if group['cls']=='car' else 40.
        excluded=('range' if raw and t['raw_ego_distance']>=limit else 'score' if mean is not None and mean<threshold else 'none')
        if raw:paint(img,raw,meta,ORANGE if excluded!='none' else CYAN)
        title='{} | {} frame {:02d} | {} | ID{}'.format(v,scene['scene'],frame,camera,tid)
        score='avg {:.6f} / threshold {:.6f} | excluded: {}'.format(mean,threshold,excluded) if mean is not None else 'No evaluated track'
        info='raw ego dist {:.3f}m | raw GT XY err {:.3f}m'.format(t['raw_ego_distance'],t['raw_actual_gt_distance']) if raw else 'No raw output; track may still be active'
        parts.append(panel(img,roi,[title,score,info,'White: actual GT | green: 3D input | red: prediction | cyan/orange: output']))
        full=out/'images'/('{}_f{:02d}_{}_full.jpg'.format(scene['scene'],frame,v))
        cv2.imwrite(str(full),img,[cv2.IMWRITE_JPEG_QUALITY,92])
        manifest.append(dict(scene=scene['scene'],frame=frame,variant=v,camera=camera,source=str(src),
            sample_token=r['sample_token'],sample_data_token=meta['sample_data_token'],camera_timestamp=meta['timestamp'],
            roi=roi,mean_score=mean,threshold=threshold,excluded=excluded,full_image=str(full.relative_to(out))))
    image=np.hstack(parts);path=out/'images'/('{}_f{:02d}_compare.jpg'.format(scene['scene'],frame))
    cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,94])
    return image,manifest


def render_mid(cfg,out,archive):
    name='scene-0275';frame=5;camera='CAM_BACK';ci=USED_ALLCAM.index(camera)
    root=output_root(cfg);step=next(s for s in load_pickle(root/'inputs'/(name+'.pkl'))['steps'] if s['frame_idx']==frame and not s['data']['is_key_frame'])
    stamp=step['camera_times'][camera];files=list((resolve(cfg['dataset_root'])/'sweeps'/camera).glob('*__{}.jpg'.format(stamp)))
    assert len(files)==1,files
    original=cv2.imread(str(files[0]));assert original is not None
    details=read_json(out/'mid_0275.json');det=np.asarray(step['data']['single_2d_info'][camera])
    roi=[875,390,1195,573];parts=[];proof={}
    config=yaml.safe_load(resolve(cfg['tracker_config']).read_text())
    for v in ('B0','B1'):
        img=original.copy();r=next(r for r in load_pickle(out/(name+'_'+v+'_trace.pkl'))['records'] if r['frame_idx']==5 and not r['is_keyframe'])
        p=np.asarray(r['predictions']);ids=p[:,14].astype(int).tolist()
        projection=fast_project_dets_to_images(step['data']['trans_infos'],p[:,:14])['np_dets_bbox2d'][ci]
        fc={'basic':{'CLASS_NUM':7},'fusion':{'fusion_algorithm':'Hungarian',
            'fusion_first_thre':config['association']['high_thre'],'fusion_metrics':'iou_2d'}}
        fusion=datafusion(np.c_[projection,p[:,12:14]],det,fc);cost=fusion.compute_cost();mi,di,_=fusion.data_association()
        mapping={ids[i]:int(d) for i,d in zip(mi,di)}
        assert mapping[15]==details[v]['selected_detection']
        for tid,color in [(15,RED),(65,PURPLE)]:
            box=projection[ids.index(tid)];cv2.rectangle(img,tuple(box[:2].astype(int)),tuple(box[2:].astype(int)),color,2)
            label_xy=(int(box[0])-48,int(box[1])-20) if tid==15 else (int(box[2])+8,int(box[1])-4)
            draw_text(img,'ID'+str(tid),label_xy,color)
        for d,color in [(10,GREEN),(21,ORANGE)]:
            box=det[d,:4];cv2.rectangle(img,tuple(box[:2].astype(int)),tuple(box[2:].astype(int)),color,2)
        chosen=details[v]['detection'];post=details[v]['post_score']
        parts.append(panel(img,roi,[v+' | scene-0275 MID before 05 | CAM_BACK',
            'ID15 <- det{} score {:.3f} | ID65 <- det{}'.format(mapping[15],chosen[-2],mapping[65]),
            'ID15 post-mid track score {:.6f}'.format(post),
            'Red ID15 | purple ID65 | green det10 (0.827) | orange det21 (0.115)']))
        proof[v]=dict(camera_source=str(files[0]),camera_timestamp=stamp,batch_timestamp=step['timestamp_us'],
            assignment={str(t):mapping[t] for t in (15,65)},
            cost_1_minus_iou={str(t):{str(d):float(cost[4,ids.index(t),d]) for d in (10,21)} for t in (15,65)},
            limit=config['association']['high_thre'][4])
        cv2.imwrite(str(out/'images'/('scene-0275_mid05_'+v+'_full.jpg')),img,[cv2.IMWRITE_JPEG_QUALITY,92])
    image=np.hstack(parts);cv2.imwrite(str(out/'images/scene-0275_mid05_compare.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,94])
    write_json(out/'mid_assignment_verified.json',proof)
    return image


def main():
    cfg=load_config();out=output_root(cfg)/'fn_inspection';(out/'images').mkdir(exist_ok=True)
    archive=load_pickle(out/'official_selected.pkl');cases=read_json(out/'evidence.json');manifest=[]
    frames={'scene-0096':[19,20],'scene-0274':[7,10],'scene-0275':[0,5]}
    for scene in cases:
        images=[]
        for frame in frames[scene['scene']]:
            image,items=render_key(cfg,out,scene,frame,archive);images.append(image);manifest.extend(items)
        if scene['scene']=='scene-0275':images.append(render_mid(cfg,out,archive))
        path=out/'images'/(scene['scene']+'_overview.jpg')
        cv2.imwrite(str(path),np.vstack(images),[cv2.IMWRITE_JPEG_QUALITY,94])
        print('RENDERED',path,flush=True)
    write_json(out/'render_manifest.json',manifest)


if __name__=='__main__':main()
