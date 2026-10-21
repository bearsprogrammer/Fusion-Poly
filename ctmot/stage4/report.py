"""Tabular, plotted and written ablation findings; includes negative outcomes."""
from collections import defaultdict
import numpy as np
from ctmot.proto.association_replay import write_csv
from .common import *

METRICS=['amota','amotp','mota','ids','frag','fp','fn']


def generate(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=output_root(cfg);out=root/'report';out.mkdir(exist_ok=True)
    local=read_json(root/'multicase/report.json')
    full={v:read_json(root/'evaluation'/v/'metrics_summary.json') for v in ['B0','B1','B2','B3']}
    runtime=read_json(root/'tracking_full.json')
    metric_rows=[]
    for v,s in full.items():
        for cls in ['overall']+list(s['label_metrics']['amota']):
            metric_rows.append(dict(variant=v,class_name=cls,**{m:s[m] if cls=='overall' else s['label_metrics'][m][cls] for m in METRICS}))
    write_csv(out/'metrics.csv',metric_rows)
    write_csv(out/'runtime.csv',[dict(variant=v,**r) for v,r in runtime['variants'].items()])
    scene_names={s['token']:s['name'] for s in read_json(resolve(cfg['dataset_root'])/'v1.0-trainval/scene.json')}
    scene_rows=[];aggregates={}
    for v in full:
        rows=read_json(root/'evaluation'/v/'scene_metrics.json')
        aggregate=defaultdict(lambda:dict(ids=0,frag=0,fp=0,fn=0,gt=0))
        for row in rows:
            for group in ['overall']+(['car'] if row['class_name']=='car' else []):
                key=(row['policy'],row['scene_token'],group)
                for k,source in [('ids','num_switches'),('frag','num_fragmentations'),('fp','num_false_positives'),('fn','num_misses'),('gt','num_objects')]:
                    value=row[source]
                    if isinstance(value,(int,float)): aggregate[key][k]+=value
        aggregates[v]=aggregate
    for v in ['B1','B2','B3']:
        for (policy,token,group),a in aggregates[v].items():
            b=aggregates['B0'][(policy,token,group)]
            delta={k:a[k]-b[k] for k in ('ids','frag','fp','fn')}
            errors=delta['ids']+delta['fp']+delta['fn']
            scene_rows.append(dict(variant=v,scene=scene_names[token],class_name=group,policy=policy,
                classification='improved' if errors<0 else 'worsened' if errors>0 else 'unchanged',
                delta_errors=errors,**{'delta_'+k:d for k,d in delta.items()},baseline=b,current=a))
    write_csv(out/'scene_changes.csv',scene_rows)
    ablation=[]
    for a,b,purpose in [('B0','B1','timestamp mean'),('B1','B2','motion center model'),('B2','B3','past acceleration')]:
        for cls in ['overall','car']:
            av=next(r for r in metric_rows if r['variant']==a and r['class_name']==cls)
            bv=next(r for r in metric_rows if r['variant']==b and r['class_name']==cls)
            ablation.append(dict(comparison=a+'->'+b,class_name=cls,purpose=purpose,**{m:bv[m]-av[m] for m in METRICS}))
    write_csv(out/'ablation.csv',ablation)
    write_json(out/'results.json',dict(metrics=metric_rows,runtime=runtime,multicase=local['summary'],
                                      ablation=ablation,scene_changes=scene_rows))
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    colors=['#686868','#d99730','#247ba5','#249975']
    for ax,m in zip(axes,['amota','ids','fn']):
        x=np.arange(4);overall=[full[v][m] for v in full];car=[full[v]['label_metrics'][m]['car'] for v in full]
        ax.bar(x-.18,overall,.36,label='Overall',color=colors)
        ax.bar(x+.18,car,.36,label='Car',color=colors,alpha=.4,hatch='//')
        ax.set_xticks(x);ax.set_xticklabels(list(full));ax.set_title(m.upper());ax.legend()
    fig.suptitle('150-scene official nuScenes val | prediction-mean ablation')
    fig.tight_layout();fig.savefig(str(out/'full_metrics.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));pairs=['B0->B1','B1->B2','B2->B3'];x=np.arange(3)
    for ax,metric,scale in zip(axes,['amota','ids'],[100.,1.]):
        for j,(cls,color) in enumerate([('overall','#247ba5'),('car','#d99730')]):
            vals=[next(r[metric] for r in ablation if r['comparison']==pair and r['class_name']==cls)*scale for pair in pairs]
            positions=x+(j-.5)*.36
            ax.bar(positions,vals,.36,color=color,label=cls.capitalize())
            for pos,val in zip(positions,vals):
                ax.annotate(('{:+.3f}' if metric=='amota' else '{:+.0f}').format(val),
                    (pos,val),xytext=(0,5 if val>=0 else -5),textcoords='offset points',
                    ha='center',va='bottom' if val>=0 else 'top',fontsize=9)
        ax.axhline(0,color='black',linewidth=.8);ax.margins(y=.25)
        ax.set_xticks(x);ax.set_xticklabels(['B1 - B0\nActual time','B2 - B1\nCV center','B3 - B2\nAcceleration'])
        ax.set_ylabel('AMOTA change (percentage points; higher is better)' if metric=='amota' else 'IDS change (lower is better)')
        ax.legend()
    fig.suptitle('Official full-val ablation | differences between adjacent variants')
    fig.tight_layout();fig.savefig(str(out/'ablation_delta.png'),dpi=160);plt.close(fig)
    ids=[c['event_id'] for c in cfg['cases']];variants=['baseline','actual_dt_ctrv','ct_cv','ct_ca','past_xyz_cv']
    matrix=np.array([[next(r['continuity'] for r in local['summary'] if r['event_id']==i and r['variant']==v) for i in ids] for v in variants])
    fig,ax=plt.subplots(figsize=(10,4.5));ax.imshow(matrix,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto',alpha=.35)
    for y in range(5):
        for x in range(5):
            r=next(r for r in local['summary'] if r['event_id']==ids[x] and r['variant']==variants[y])
            ax.text(x,y,('Recovered' if matrix[y,x] else 'Not recovered')+('\nCA→CV fallback' if r['ca_fallback_cv'] else ''),ha='center',va='center',fontsize=9)
    ax.set_xticks(range(5));ax.set_xticklabels([c['scene']+'\nID'+str(c['track_id'])+' / '+c['mechanism'] for c in cfg['cases']])
    ax.set_yticks(range(5));ax.set_yticklabels(['B0','B1','B2','B3','B4 diagnostic'])
    ax.set_title('Local one-shot replay: correct association and subsequent ID continuity')
    fig.tight_layout();fig.savefig(str(out/'multicase_recovery.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    palette=colors+['#8752a1'];x=np.arange(len(ids));width=.16
    for j,v in enumerate(variants):
        selected=[next(r for r in local['summary'] if r['event_id']==i and r['variant']==v) for i in ids]
        positions=x+(j-2)*width
        axes[0].bar(positions,[r['prediction_xy_error'] for r in selected],width,
                    color=palette[j],label='B'+str(j)+(' diagnostic' if j==4 else ''))
        axes[1].bar(positions,[r['cost'] for r in selected],width,color=palette[j])
        for pos,r in zip(positions,selected):
            if not r['gate_pass']:
                axes[1].plot(pos,r['cost']+.025,'kx',markersize=6)
    axes[0].set_ylabel('Prediction XY error to GT (m)')
    axes[0].legend(ncol=5,fontsize=9)
    axes[1].axhline(1.1,color='black',linestyle='--',linewidth=1,label='Car cost limit 1.1')
    axes[1].plot([],[],'kx',label='Rejected: spatial or cost gate')
    axes[1].legend(fontsize=9)
    axes[1].set_ylabel('Correct-detection geometry cost')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c['scene']+'\nID'+str(c['track_id'])+' / '+c['mechanism'] for c in cfg['cases']])
    fig.suptitle('Local replay | identical posterior, observations and association settings')
    fig.tight_layout();fig.savefig(str(out/'multicase_prediction.png'),dpi=160);plt.close(fig)
    lines=['# Stage 4 results','',
        '150개 val scene의 공식 지표입니다. B0–B3는 전체 track에 동일 정책을 적용하며 GT/대상 ID 개입이 없습니다.',
        '각 variant는 keyframe 6,019개와 mid-camera 묶음 5,869개(총 11,888 event)를 처리했습니다. High 입력을 사용하되 공식 tracking 채점은 keyframe 결과에 대해 수행합니다.',
        '기존 KF의 state 표현·yaw·공분산을 유지한 prediction mean 실험으로, 완전한 CT filter 교체와는 구분해야 합니다.','',
        '| Variant | Overall AMOTA | Car AMOTA | Overall IDS | Car IDS | Overall FN | Car FN |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v,s in full.items():
        lines.append('| {} | {:.6f} | {:.6f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} |'.format(v,s['amota'],s['label_metrics']['amota']['car'],s['ids'],s['label_metrics']['ids']['car'],s['fn'],s['label_metrics']['fn']['car']))
    lines+=['','## 전체 공식 지표','',
        '| Variant | Class | AMOTA ↑ | AMOTP ↓ | MOTA ↑ | IDS ↓ | Frag ↓ | FP ↓ | FN ↓ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in metric_rows:
        lines.append('| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} |'.format(
            r['variant'],r['class_name'],*[r[m] for m in METRICS]))
    lines+=['','## Local replay','',
        '| Scene / old ID | 분류 | B0 | B1 | B2 | B3 | B4 diagnostic |',
        '|---|---|---|---|---|---|---|']
    for c in cfg['cases']:
        cells=[]
        for v in variants:
            r=next(r for r in local['summary'] if r['event_id']==c['event_id'] and r['variant']==v)
            cells.append(('복구' if r['continuity'] else '실패')+(' (CV fallback)' if r['ca_fallback_cv'] else ''))
        lines.append('| {} / {} | {} | {} |'.format(c['scene'],c['track_id'],c['mechanism'],' | '.join(cells)))
    lines+=['',
        '복구는 정답 detection의 실제 할당과 query 이후 2개 keyframe까지 동일 ID 유지로 판정했습니다. scene-1072의 Stage 3 비용·continuity를 재현했고, local 구간의 다른 객체 lineage assignment/state/lifecycle 변경은 없었습니다.',
        '- 0099-ID45: B1은 cost 1.0800이지만 detection까지 XYZ 거리 3.2938m로 spatial gate를 벗어났습니다. B2는 2.9795m, B3는 2.5569m로 복구합니다. B2의 gate 여유는 약 2cm입니다.',
        '- 0101-ID14: CA는 XY error를 5.5214m에서 2.9688m로 줄였지만 cost 1.1771 > 1.1로 복구하지 못했습니다.',
        '- 0915-ID108: CV의 detection 거리는 3.9604m로 gate 밖입니다. CA는 accepted history 부족으로 CV fallback이므로 충분한 history를 쓴 CA 결과가 아닙니다.',
        '- P2 두 사례의 이전 track은 재등장 시 active였습니다. 이 실패를 lifecycle 종료로 분류하지 않습니다.']
    lines+=['','## 실행 시간','',
        '| Variant | Tracking 시간 합 (s) | Keyframe FPS | Event FPS | 전처리 포함 keyframe FPS | 평가 시간 (s) |',
        '|---|---:|---:|---:|---:|---:|']
    for v,r in runtime['variants'].items():
        seconds=read_json(root/'evaluation'/v/'provenance.json')['seconds']
        lines.append('| {} | {:.2f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |'.format(
            v,r['tracking_cpu_wall_sum_s'],r['keyframe_fps'],r['event_fps'],r['end_to_end_keyframe_fps'],seconds))
    counts=runtime['variants']['B3']['counters']
    lines+=['','B3의 CA 적용은 {:,}/{:,}회({:.2%}), CV fallback은 {:,}회입니다.'.format(
        counts['ca_predictions'],counts['predictions'],counts['ca_predictions']/counts['predictions'],counts['ca_fallback_cv']),
        '시간 합은 6개 worker의 scene별 실측값을 더한 값이며, 단독 실행 벤치마크 또는 병렬 batch FPS가 아닙니다. Full tracking의 batch wall-clock은 병합 중단으로 null이며 추정값을 채우지 않았습니다.']
    lines+=['','## Ablation','', '| Comparison | Class | ΔAMOTA | ΔIDS | ΔFP | ΔFN |','|---|---|---:|---:|---:|---:|']
    for r in ablation:
        lines.append('| {} | {} | {:+.6f} | {:+.0f} | {:+.0f} | {:+.0f} |'.format(r['comparison'],r['class_name'],r['amota'],r['ids'],r['fp'],r['fn']))
    lines+=['','## Scene 분류','',
        '주 비교는 B0의 class별 best-MOTA score threshold를 모든 variant에 공통 적용했습니다. scene별 Δ(FN+FP+IDS)<0이면 improved, >0이면 worsened입니다. 공식 summary 자체는 각 variant의 best-MOTA threshold를 사용하므로 두 결과를 혼동하지 않습니다.','']
    for v in ['B1','B2','B3']:
        selected=[r for r in scene_rows if r['variant']==v and r['policy']=='common_b0' and r['class_name']=='overall']
        counts={k:sum(r['classification']==k for r in selected) for k in ['improved','worsened','unchanged']}
        lines.append('- {}: {}.'.format(v,counts))
        for label,items in [('가장 개선된 scene',sorted(selected,key=lambda r:r['delta_errors'])[:5]),
                            ('가장 악화된 scene',sorted(selected,key=lambda r:r['delta_errors'],reverse=True)[:5])]:
            lines.append('  - {}: {}.'.format(label,', '.join('{} ({:+.0f})'.format(r['scene'],r['delta_errors']) for r in items)))
    lines+=['','## 한계','',
        '- Multi-case는 한 번의 개입 후 원본 tracker를 계속 실행합니다. Full run은 매 예측 단계에 정책을 적용하므로 local 숫자와 직접 동일시하지 않습니다.',
        '- B1은 car CTRV를 포함한 class별 원래 model의 평균 전파에 actual Δt를 적용합니다. B2/B3는 같은 actual-dt yaw를 유지하고 geometric center만 ODE로 대체합니다.',
        '- covariance는 모든 variant에서 원본 nominal schedule/F/Q를 유지합니다. 동적으로 달라진 posterior로 인한 공분산 차이는 정상적인 후속 효과입니다.',
        '- CT-CV/CA는 마지막 실제 3D correction 또는 birth의 posterior 위치·물리적 XY 속도를 시작점으로 합니다. vz=az=0이며 원래 Z 고정 정책을 유지합니다.',
        '- CA는 최근 최대 4개의 accepted 3D detection velocity를 실제 timestamp로 OLS 적합합니다. birth는 가속도 history에서 제외하며 3개 미만이면 CV입니다. clipping/tuning 없이 고정 설정입니다.',
        '- camera-only 단계는 기존 코드상 motion correction이 없으므로 CT anchor를 갱신하지 않습니다. 모든 camera 관측을 사용할 수 있는 latest-camera timestamp를 그룹 처리 시각으로 사용합니다.',
        '- FPS는 cached detection 입력에서 tracker.advance() 시간 합으로 계산합니다. 병렬 worker 간 시간 합이며 평가 시간·cache 읽기·초기화·deepcopy·최종 JSON 저장은 제외합니다. 전처리를 더한 FPS도 runtime.csv에 있지만 시스템 전체 end-to-end 속도는 아닙니다.',
        '- 5개 사례는 이미 실패한 event를 선택한 표본입니다. 전체 val 지표가 일반화 판단의 기준입니다.', '']
    lines+=['- CV/CA ODE는 해당 등속/등가속도 모델의 해석해와 동등합니다. 개선이 있더라도 ODE 수치 적분 자체의 우월성이나 inter-frame 정보 손실 가설을 증명하는 것은 아닙니다. 이번 실험은 timestamp와 motion prior의 영향을 분리하며, sweep 기반 새 관측이나 CT association cost는 추가하지 않았습니다.',
        '- 단일 고정 설정의 val 비교입니다. 작은 지표 차이를 통계적 유의성 또는 일반적인 성능 보장으로 해석하지 않습니다.','']
    lines+=['## 그림 및 상세 데이터','',
        '- [전체 공식 지표 그래프](full_metrics.png)',
        '- [Ablation 변화량 그래프](ablation_delta.png): AMOTA는 percentage point, IDS는 개수 변화입니다.',
        '- [Local prediction error / association cost](multicase_prediction.png): × 표시는 spatial 또는 cost gate 탈락입니다. Cost가 1.1보다 작아도 3m spatial gate를 벗어나면 실패합니다.',
        '- [Local ID continuity 복구](multicase_recovery.png)',
        '- [공식 전체·class별 지표 CSV](metrics.csv), [ablation CSV](ablation.csv), [scene별 변화 CSV](scene_changes.csv), [runtime CSV](runtime.csv)',
        '- [Local 상세 CSV](../multicase/comparison.csv), [local 요약 CSV](../multicase/summary.csv)','']
    for v in ['B2','B3']:
        change=full[v]['amota']-full['B1']['amota']
        lines.append('- {} vs B1: overall AMOTA {:+.6f}. {}'.format(v,change,
            '이번 고정 설정에서 추가 AMOTA 이점이 관측됐습니다.' if change>0 else '이번 고정 설정에서는 추가 AMOTA 이점이 관측되지 않았습니다.'))
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('REPORT',out,flush=True)
