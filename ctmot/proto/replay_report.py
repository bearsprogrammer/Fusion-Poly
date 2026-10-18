"""Static, exportable local replay diagnostics."""
import numpy as np


def render_report(output, report, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows=report['evaluation']['rows']; summary=report['evaluation']['summary']
    names=list(summary); frame=report['case']['query_frames'][0]
    labels=['B0','Actual-dt CTRV','CT-CV','CT-CA','Past XYZ CV']
    colors=['#b83a38','#d89631','#237ab5','#259c77','#8257ac']
    selected=[next(r for r in rows if r['variant']==name and r['frame_idx']==frame) for name in names]
    fig,axes=plt.subplots(1,3,figsize=(18,5.2))
    for j,(row,color) in enumerate(zip(selected,colors)):
        axes[0].bar(j,row['raw_geometry_cost'],color=color)
        axes[0].text(j,row['raw_geometry_cost']+.025,'{:.4f}\n{}'.format(row['raw_geometry_cost'], 'MATCH' if row['correct_detection_assigned_to_target'] else 'REJECT'),ha='center',fontsize=9,
                     bbox=dict(facecolor='white',edgecolor='none',pad=1),zorder=4)
    axes[0].axhline(selected[0]['cost_limit'],color='black',ls='--',label='Original car cost limit')
    axes[0].set_xticks(range(len(names)));axes[0].set_xticklabels(labels,rotation=20,ha='right')
    axes[0].set_ylabel('1 - fast GIoU3D');axes[0].set_ylim(0,max(r['raw_geometry_cost'] for r in selected)+.28)
    axes[0].set_title('Frame {}: actual cascade assignment'.format(frame));axes[0].legend(fontsize=8)
    frames=sorted({r['frame_idx'] for r in rows})
    table=np.array([[next(r['correct_owner_id'] for r in rows if r['variant']==name and r['frame_idx']==f) for f in frames] for name in names])
    axes[1].imshow(table==report['case']['track_id'],vmin=0,vmax=1,cmap='RdYlGn',aspect='auto',alpha=.35)
    for y in range(len(names)):
        for x in range(len(frames)):
            row=next(r for r in rows if r['variant']==names[y] and r['frame_idx']==frames[x])
            axes[1].text(x,y,'{}{}'.format(table[y,x], '*' if row['owner_kind']=='birth' else ''),ha='center',va='center')
    axes[1].set_xticks(range(len(frames)));axes[1].set_xticklabels(frames)
    axes[1].set_yticks(range(len(names)));axes[1].set_yticklabels(labels)
    axes[1].set_xlabel('Keyframe (0-based); * = new track birth')
    axes[1].set_title('ID receiving the target detection')
    for name,label,color in zip(names,labels,colors):
        seq=[r for r in rows if r['variant']==name]
        axes[2].plot([r['frame_idx'] for r in seq],[r['prediction_gt_xy_m'] for r in seq],'-o',label=label,color=color)
    axes[2].set_ylabel('ID{} pre-association GT XY error (m)'.format(report['case']['track_id']))
    axes[2].set_xlabel('Keyframe (0-based)');axes[2].set_xticks(frames)
    axes[2].set_title('One intervention, then original tracker');axes[2].legend(fontsize=8)
    fig.suptitle('{} / ID{} | high, val | local replay (not official IDS evaluation)'.format(report['case']['scene'],report['case']['track_id']))
    fig.tight_layout();fig.savefig(str(output/'replay_comparison.png'),dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,7))
    origin=np.array(report['plans']['ct_cv']['initial_state_pv'][:2])
    for row,label,color in zip(selected,labels,colors):
        p=np.array(row['prediction_xyz'][:2])-origin
        ax.scatter(*p,label=label,color=color,s=80,marker='x')
    gt=np.array(selected[0]['gt_xyz'][:2])-origin
    candidate=report['evaluation']['evaluation_candidates'][frame]['detection']
    d=np.array(candidate[:2])-origin
    ax.scatter(*gt,label='GT',color='black',marker='*',s=130)
    ax.scatter(*d,label='Correct aligned detection',facecolors='none',edgecolors='green',s=130)
    ax.scatter(0,0,label='Common KF posterior at anchor',color='gray',s=60)
    ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=9)
    ax.set_xlabel('Global X offset (m)');ax.set_ylabel('Global Y offset (m)')
    ax.set_title('Frame {} prediction centers; Z differs in Past XYZ CV'.format(frame))
    fig.tight_layout();fig.savefig(str(output/'prediction_centers.png'),dpi=160);plt.close(fig)

    lines=['# CT association replay', '',
        '- scene 시작부터 baseline을 복원하고, anchor posterior에서 분기했습니다.',
        '- query keyframe에서 한 번만 예측 평균을 교체한 뒤 원래 tracker를 계속 실행했습니다.',
        '- CT-CV/CA/Past XYZ CV: 중심 XYZ만 KF 내부 평균과 box/cache에 일관되게 반영했습니다. 원래 speed/yaw/yaw-rate와 공분산은 유지합니다.',
        '- Actual-dt CTRV: 실제 Δt로 CTRV 평균을 전파합니다. 공분산은 원래 값을 유지하므로 완전한 actual-dt EKF 실험은 아닙니다.',
        '- 검출, Hungarian, mix→pure3D→2D cascade, gate, score, lifecycle 설정은 변경하지 않았습니다.',
        '- GT는 replay 완료 후 올바른 검출과 오차를 평가할 때만 사용했습니다.',
        '- 아래 연속성은 올바른 검출을 받는 ID 기준이며, 공식 nuScenes IDS 재평가는 아닙니다.', '',
        '| Variant | query cost | Correct assigned | Owner IDs (window) | Continuity |',
        '|---|---:|---|---|---|']
    for name,s in summary.items():
        lines.append('| {} | {:.6f} | {} | {} | {} |'.format(name,s['intervention_cost'],s['intervention_selected_correct'],s['correct_owner_ids'],s['id_continuity_through_window']))
    lines += ['', '## 초기 조건 수치 비교', '',
              '| Variant | p0 XYZ (m) | v0 XYZ (m/s) | prediction XYZ (m) |',
              '|---|---|---|---|']
    for name in ('ct_cv', 'ct_ca', 'past_xyz_cv'):
        p = report['plans'][name]
        fmt = lambda values: ', '.join('{:.6f}'.format(v) for v in values)
        lines.append('| {} | {} | {} | {} |'.format(name, fmt(p['initial_state_pv'][:3]),
                      fmt(p['initial_state_pv'][3:]), fmt(p['position_m'])))
    lines+=['','## 시간 및 중간 관측', '',
        '- Nominal elapsed: {:.6f} s; actual elapsed: {:.6f} s.'.format(report['plans']['baseline']['nominal_baseline_elapsed_s'],report['plans']['ct_cv']['actual_dt_s']),
        '- 중간 camera별 실제 timestamp와 target match/update 여부: report.json의 timing.',
        '- input hash 및 baseline 수치 재현 검증: report.json.', '',
        '## 과거 XYZ CV와 현재 CT-CV', '',
        '- 과거 XYZ CV는 마지막 accepted aligned detection 위치를 시작점으로 사용하고, 두 과거 검출 위치의 실제 시간 차분으로 XYZ 속도를 계산합니다.',
        '- 현재 CT-CV는 KF posterior 위치 및 CTRV 내부 speed/yaw를 XY 속도로 변환한 값을 사용하며 Z는 고정합니다.',
        '- 두 방식의 query timestamp와 cost 함수는 같지만, 시작 위치·속도 source·Z motion이 달라 비용 차이를 CT 자체의 효과로 해석할 수 없습니다.', '',
        '## 다른 객체 영향', '',
        '- report.json의 evaluation.impacts는 모든 검출 할당을 비교합니다.',
        '- lineage는 birth sample token/stage/detection index입니다. 새 ID 생성 감소로 발생하는 숫자 이동과 실제 객체 할당 변화를 구분합니다.',
        '- frame 20 competing_tracks는 올바른 검출에 대해 원래 공간 gate를 통과한 다른 track의 실제 matrix cost입니다.', '',
        '## 범위', '',
        '- 한 scene의 한 target에 대한 국소 개입입니다. 전체 val 성능 및 CT 모델의 일반적 우월성을 증명하지 않습니다.',
        '- 후속 프레임에 CT 예측을 다시 주입하지 않습니다. CT 속도/공분산을 포함하는 완전한 필터 대체는 별도 실험입니다.', '']
    for name, s in summary.items():
        lines.append('- {}: 다른 검출의 lineage 변경 {}건; 다른 공통 lineage의 state/score/lifecycle/covariance 변경 {}건.'.format(
            name, s['unrelated_detection_lineage_changes'], s['other_shared_lineage_state_changes']))
    (output/'report.md').write_text('\n'.join(lines))
