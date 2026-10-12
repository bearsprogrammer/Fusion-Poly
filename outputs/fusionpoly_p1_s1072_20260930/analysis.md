# P1-S1072-C-102 — high / val 분석

본 사례는 **P1: 정상 검출의 association 거절**입니다. 아래 t는 실제 신규 ID가 생성된 frame20이며, 공식 IDS는 frame21입니다. 모든 frame index는0-based입니다.

| Attributes | frame19 /102 | frame20 /102 | frame20 /124 | frame21 /124 | 정의 / 주의 |
|---|---|---|---|---|---|
| Case ID | P1-S1072-C-102 | — | — | — | P1: 정상 3D 검출이 이어지는 association 실패. |
| Object class | Car | — | — | — | split: val / scene-1072 / high |
| GT ID | d1ecbeefbbd54baaa9d83a959f90cbad | — | — | — | GT instance_token |
| Pred. Track ID | 102 | 102 | 124 | 124 | old ID → new ID |
| Failure index / timestamp | frame:000019<br>1542799711898267 µs | frame:000020<br>1542799712447926 µs | frame:000020<br>1542799712447926 µs | frame:000021<br>1542799712947796 µs | 0-based keyframe. 실패/신규 생성: frame20. 공식 IDS(event419): frame21. |
| Analysis window | t−1 | t (association 실패) | t (신규 ID 생성) | t+1 (공식 IDS) | t = frame20. 검출 연속성 확인: frame16–25. |
| Hz | 4 Hz high: 2 Hz 3D keyframes + 중간 2D camera observations | — | — | — | 고정 dt=0.25s/step. 실제 sensor timestamp는 불규칙. |
| Failure mechanism | Association-only / correct-detection rejection (P1에 해당) | — | — | — | 정상 검출이 cost gate에서 거절되어 새 ID가 생성된 사건. |
| Scene event | Oncoming pass / 근접 통과 / camera handoff (front → right → back) | — | — | — | 반대 방향 차량 통과가 RGB로 확인됨. 완전 occlusion을 주원인으로 단정하지 않음. |
| 2D/3D detection | 2D + 3D 모두 존재. 중간 frame의 true 2D detection도 있으나 ID102와 연결 실패. | — | — | — | 4Hz를 4Hz 3D detector 입력으로 해석하지 않음. |
| Detection quality | Raw det–GT 3D IoU: 0.8024<br>Aligned det–GT 3D IoU: 0.8024<br>Raw XY error: 0.068 m | Raw det–GT 3D IoU: 0.8498<br>Aligned det–GT 3D IoU: 0.8557<br>Raw XY error: 0.049 m | Raw det–GT 3D IoU: 0.8498<br>Aligned det–GT 3D IoU: 0.8557<br>Raw XY error: 0.049 m | Raw det–GT 3D IoU: 0.7845<br>Aligned det–GT 3D IoU: 0.8036<br>Raw XY error: 0.040 m | 정확한 oriented cuboid IoU (roll/pitch 포함). GT frame 시각 기준. Raw와 GAAM aligned를 구분. |
| Track prediction | center (680.556, 2134.544, 2.366) m<br>vxy (5.655, 9.128) m/s<br>speed 10.738 m/s<br>yaw 58.219° | center (683.905, 2138.558, 2.007) m<br>vxy (6.448, 8.132) m/s<br>speed 10.379 m/s<br>yaw 51.589° | Newborn: 사전 prediction 없음<br>초기화 center (684.952, 2140.069, 1.158) m<br>초기화는 mix[0] 관측에서 수행 | center (688.577, 2145.228, 1.158) m<br>vxy (7.250, 10.319) m/s<br>speed 12.611 m/s<br>yaw 54.908° | association 직전 global 좌표. CTRV state 기반 vx=v cos(yaw), vy=v sin(yaw). z가 포함된 center. |
| Correct candidate | ACCEPT: mix[0]<br>raw[0] / score 0.8348 | REJECT: mix[0]<br>raw[0] / score 0.8396 | NEW BIRTH: mix[0]<br>raw[0] / score 0.8396 | ACCEPT: mix[3]<br>raw[1] / score 0.7736 | GT에 대응하는 실제 3D detection. mix index는 frame별 로컬 인덱스. |
| Selected candidate | mix[0] = correct detection | None (unmatched) | mix[0]으로 초기화<br>기존 track과의 assignment 아님 | mix[3] = correct detection | 배당 관측이 없으면 IoU와 selected cost도 N/A로 구분. |
| Selected-detection IoU | 3D 0.2672 / BEV 0.5590 | N/A (no assigned detection) | N/A (newborn) | 3D 0.4966 / BEV 0.8176 | pre-update prediction 대 assigned detection. frame20 old ID에는 assigned detection 없음. |
| Association cost | 0.7521132020645214 | 1.1473571838075898 | N/A (newborn) | 0.49547656983493316 | 실제 Hungarian matrix의 true-candidate 항. cost=1−GIoU3D, car limit=1.1. frame20의 selected cost는 N/A. |
| Association margin | N/A (finite candidate 1개) | N/A (finite candidate 1개) | N/A (newborn) | N/A (finite candidate 1개) | best vs second-best finite cost. 다른 car는 3m XYZ spatial gate 밖 → +∞. margin=0으로 처리하지 않음. |
| Observation gap | 0 ms / 0 keyframes | 0 ms / 0 keyframes | 0 ms / 0 keyframes | 0 ms / 0 keyframes | 3D detector miss 기준. frame16–25 연속 검출. frame19→20의 549.659ms는 샘플 간격이며 miss가 아님. |
| Minimum inter-object distance | 7.8358885265168485 | 5.737848028660238 | 5.737848028660238 | 15.419359584626353 | 해당 시각의 두 GT center XY 거리(m). 상대 GT=9c8c9fc0a03d462e8d58cf6bb01962be. window18–22 실측 최소 5.738m; 선형 보간 최소 3.443m (별도 추정). |
| Relative motion | Δxy (6.782, 3.925) m<br>Δvxy (-11.479, -17.086) m/s<br>relative speed 20.584 m/s<br>Δheading 178.901°<br>target 12.003, ego 8.871 m/s | Δxy (0.300, -5.730) m<br>Δvxy (-11.806, -17.425) m/s<br>relative speed 21.048 m/s<br>Δheading 179.183°<br>target 12.726, ego 8.764 m/s | Δxy (0.300, -5.730) m<br>Δvxy (-11.806, -17.425) m/s<br>relative speed 21.048 m/s<br>Δheading 179.183°<br>target 12.726, ego 8.764 m/s | Δxy (-5.609, -14.363) m<br>Δvxy (-12.122, -17.772) m/s<br>relative speed 21.513 m/s<br>Δheading 179.439°<br>target 13.519, ego 8.479 m/s | other−target, global XY. 같은 물리적 상대 차량을 유지. frame20 상대 차량은 유효 association 후보가 아님. |
| Inter-frame evidence | 실제 mid RGB에서 전방→우측→후방 통과 확인. frame20 직전 2D score: FRONT 0.9992, FRONT_RIGHT 0.8790. ID102와 cost: 0.6877 / 0.8236 > 0.1 → 모두 거절. frame18–22 LiDAR 39 sweeps에 target GT-box 내 points가 존재(8–1891). | — | — | — | mid GT box는 SDK 보간. raw sweep returns는 독립 3D detection/association 성공을 의미하지 않음. 완전 occlusion 원인은 미확정. |
| baseline outcome | frame20: ID102 unmatched + ID124 신규 생성. frame21: official IDS 102→124. | — | — | — | 평가 track-mean score: 102=0.635360, 124=0.718285 > car threshold 0.454122. frame20 old ID의 GT 거리 1.799m로 기존 매칭 유지 가능. |
| likely cause | 정상 3D detection을 prediction이 따라가지 못해 3D cost gate 탈락. frame20 prediction–det XY=1.838m, Δz=+0.849m. 실제 Δt=549.659ms 대비 nominal 500ms와 속도/높이 변화가 기여할 가능성. 중간 2D association도 엄격한 IoU gate에서 실패. | — | — | — | 관측된 failure chain과 원인 가설을 구분. 두 유효 후보 사이 cost 경합이나 ID 교환의 증거는 없음. |
| Causal confidence | Medium | — | — | — | 배정 거절/신규 생성은 exact replay로 확인. 시간 표현의 단독 인과성과 full-run IDS 감소는 미검증. |
| Simple CT counterfactual | ① 같은 frame19 corrected state에서 실제 Δt로 CTRV 전파: cost 1.1474 → 1.0380.<br>② 과거 accepted det18/19만 사용한 XYZ CV + 실제 Δt: cost 0.3182. size/yaw는 baseline 유지. | — | — | — | frame20의 실제 전체 stage-one Hungarian matrix에서 ID102 column만 교체. 미래 GT로 fitting하지 않음. 추가 CT cost term이나 학습 모델 실험은 아님. |
| CT association result | Corrected locally: 두 실험 모두 ID102 ← correct mix[0] 배정 성공. | — | — | — | 단일 step 반사실 실험. 이후 trajectory / official IDS / AMOTA 개선은 평가하지 않음. |
| Figure assets | case_diagnostics.png<br>CAM_FRONT_contact.jpg / CAM_FRONT_RIGHT_contact.jpg / CAM_BACK_RIGHT_contact.jpg | — | — | — | 결과 폴더에 포함. 추가 raw sweep·camera token 및 trace는 정량 분석 폴더에 보존. |
| Use in paper | Supplementary candidate: prediction/gating failure under irregular timing | — | — | — | P1 prediction/gating 실패 사례. 두 유효 후보 사이 association ambiguity를 주장할 근거는 부족. |

## 재현 및 출처

- **Results**: Fusion_Poly_EXP/result/nusc_config_high/results.json
- **SHA256**: d02fc54ac5611d548250603933798b7ae93aae40f0fa2b76dafff3ccd43f0ea5
- **Configuration**: Fusion_Poly_EXP/eval_results/nusc_config_high/config.yaml
- **Official IDS**: Fusion_Poly_EXP/inspection/nusc_config_high/ids_events.json / event419
- **3D detections**: data/detector/val/nuscenes_val_centerpoint_3d.json
- **2D detections**: data/utils/cascade_rcnn_4hz/nuscenes_val_cascade_2d_4hz.json
- **Dataset**: /home/allen-5070ti/dataset/nuScenes/full_trainval / v1.0-trainval
- **Replay / measurements**: Fusion_Poly_EXP/inspection/nusc_config_high/scene1072_quantitative/trace.json; case_analysis.json
- **Algorithm**: tracking/nusc_tracker.py:401 / utils/matching.py:16 / utils/script.py:68
- **Velocity convention**: motion_module/motion_model.py CTRV.getOutputInfo: slots labelled velocity export speed,yaw; physical vx,vy reconstructed from state.
- **Validation**: Full scene40 keyframes +39 mid steps exactly reproduce saved high outputs. Cost independently recomputed; raw detections mapped to replay; same stage Hungarian baseline reproduced.
- **Observation definition**: GT-supported car detection: XY<2m and 3D IoU>0. Main-frame3D IoU0.78–0.85. No new visualization score cutoff applied.
- **Official match timing**: Frame20 old output GT XY distance1.799495m <2m, so evaluator continuity can retain ID102. ID102 absent from output21; IDS102→124 recorded there.
- **Counterfactual limits**: Local deterministic propagation + same real Hungarian matrix. Not a complete CT model, full trajectory rerun or metric improvement.
- **Spatial evidence**: Same physical other GT: 9c8c9fc0a03d462e8d58cf6bb01962be. It is outside3m prediction spatial gate at t. Interpolated minimum3.442954m uses keyframeGT and is not a raw mid measurement.