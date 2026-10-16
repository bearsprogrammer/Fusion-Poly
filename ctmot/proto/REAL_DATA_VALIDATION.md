# 실제 track history 기반 prediction 검증

이 실행기는 검증된 Fusion-Poly high trace를 읽어 동일한 posterior에서 원본
prediction, CT-CV, CT-CA를 비교합니다. 원본 tracker를 수정하거나 재실행하지 않습니다.
nuScenes 원본 GT는 예측 결과를 저장한 후 평가에만 사용합니다.

## 실행

레포 root에서:

```bash
conda activate fusionpoly-repro
python -m ctmot.proto.validate_history \
  --config ctmot/proto/config/validation_scene1072.yaml
```

기본 사례는 val scene-1072, ID102, frame19 posterior → keyframe20입니다.
프레임 번호는 0-based이며 실제 Δt는 0.549659초입니다.

출력 폴더:

```text
ctmot/proto/outputs/scene-1072_track102_anchor19/
├── predictions.json             # GT 평가 전에 저장한 초기 상태·history·가속도·prediction
├── report.json                  # 설정, 입력/코드 hash, GT, 검출 대응, 수치와 gate 결과
├── report.md                    # 읽기 쉬운 결과표
└── comparison_frame000020.png   # 위치, GT XY 오차, association cost 비교
```

출력 파일은 같은 사례를 다시 실행하면 갱신됩니다. 실험 조건별로 보존하려면
`output.root`를 바꾸세요. 생성 결과는 Git에서 제외됩니다.

## 입력과 알고리즘

1. `source.trace_root/<scene의 하이픈 제거 이름>_quantitative/trace.json`을 읽습니다.
   trace의 exact replay 검증 상태, high/val 설정, 결과 SHA256을 확인합니다.
   anchor와 query의 저장 출력도 실제 `results.json`과 직접 대조합니다.
2. `history.py:build_snapshot`이 anchor에서 correction을 마친 내부 CTRV state를
   추출합니다. `results.json`의 velocity 필드를 초기 속도로 사용하지 않습니다.
   이 CTRV 출력 경로는 `[speed, yaw]`를 내보내므로 내부 state에서
   `vx=speed*cos(yaw)`, `vy=speed*sin(yaw)`를 복원합니다.
3. 동일 ID가 anchor까지 실제 3D association에서 배정받은 관측만 선택합니다.
   신규 생성 관측, mid prediction, 미배정 keyframe prediction은 제외합니다.
   scene-1072에서는 요청한 최대 4개 중 frame17·18·19의 3개를 사용합니다.
4. CA 가속도는 실제 관측 timestamp를 사용하여 detection의 vx, vy를 각각
   시간에 대해 최소제곱 직선 회귀한 기울기입니다. 회귀 절편으로 초기 속도를
   바꾸지 않습니다. CA와 CV 모두 같은 posterior 위치·속도에서 출발합니다.
5. 현재 비교는 `vertical_policy: hold`만 지원합니다. `vz=az=0`이며 Z는 posterior
   값으로 유지합니다. 수직 운동 추정과 수평 모델 변경 효과를 섞지 않기 위한 조건입니다.
6. `predict_shadow`에는 past-only snapshot과 query의 frame/token/timestamp만
   전달됩니다. GT나 query detection은 predictor/가속도 추정기의 입력이 아닙니다.
7. B0는 원본 nominal Δt로 anchor에서 전파했을 때 기록된 prediction과 일치하는지
   검증합니다. 중간 correction으로 상태가 달라진 query나 trace step 누락은 거절합니다.
   따라서 correction을 받은 B0와 개방 루프 CT를 혼합 비교하지 않습니다.
8. prediction을 먼저 저장한 후, dataset의 sample/scene과 timestamp를 확인하고
   원본 `sample_annotation.json`에서 query GT를 읽습니다. GT instance ID는 평가에만 사용합니다.

## 평가 정의

- 위치 오차: 각 모델의 prediction과 같은 keyframe GT의 XY 거리, signed Z 차이,
  XYZ 거리. 검출과의 오차와 혼동하지 않도록 `gt_*` 이름으로 저장합니다.
- 검출 대응: query의 같은 클래스 검출 중 GT 중심에 가장 가까우며
  `detection_gt_max_xy_m` 이내인 후보를 평가용으로 선택합니다. 이는 평가의 대응 규칙이며
  원본 tracker의 association 규칙을 바꾸지 않습니다. 대응 후보가 없으면 cost/gate는 N/A입니다.
- Cost: repo의 기존 `1 - fast_giou_3d`를 사용합니다. query baseline의 박스 size/yaw를
  모든 모델에 동일하게 사용하고 중심만 교체합니다. 가속도로 yaw를 추정하지 않습니다.
- Gate: original trace의 해당 stage car cost limit과 3m XYZ spatial gate를 사용합니다.
  spatial gate는 원본처럼 거리 `< limit` 조건입니다. 두 gate 모두 통과해야 `gate_pass=true`입니다.
- B0의 재계산 cost는 원본 실제 matrix 항과 일치하는지도 검사합니다.
- Hungarian은 실행하지 않습니다. gate 통과를 배정 성공·IDS 감소로 해석하지 않습니다.

## scene-1072 실행 결과

초기 상태: `[680.826711, 2134.381927, 2.006793, 5.859264, 8.566563, 0]`.

과거 3개 관측으로 추정한 가속도: `[1.294906, -0.203206, 0]` m/s².

| 모델 | GT XY 오차 (m) | GT XYZ 오차 (m) | Cost | 두 gate 통과 |
|---|---:|---:|---:|---|
| B0 high | 1.799495 | 2.025053 | 1.147357 | 실패 |
| CT-CV | 1.291482 | 1.590780 | 1.049496 | 통과 |
| CT-CA | 1.194693 | 1.513257 | 1.020027 | 통과 |

세 모델 모두 GT 대비 Z 차이는 +0.928793m입니다. 이번 비교는 이를 보정하지 않습니다.
기존의 `actual-dt CTRV` 또는 과거 검출 위치에서 출발한 `XYZ CV` 반사실 실험과는
모델/출발 조건이 다르므로 당시 cost와 동일한 실험으로 취급하면 안 됩니다.

## 다른 scene으로 변경

config를 복사하고 `case.scene`, `track_id`, `anchor_frame`, `query_frames`,
`gt_instance_id`를 함께 변경하세요. `dataset.root`, `source.*`, `output.root`도 수정할 수 있습니다.

다음 조건을 만족해야 합니다.

- 공식 val split의 car/CTRV 사례여야 합니다. geometry-filter CTRV state는 지원하지 않습니다.
- 해당 scene의 검증된 replay trace가 있어야 합니다. `results.json`만으로 내부 posterior와
  assignment history를 복원하지 않습니다. trace가 없으면 명확한 오류를 반환합니다.
- anchor는 corrected posterior가 있어야 하며 query는 그 이후 keyframe이어야 합니다.
- trace에 필요한 과거 history와 중간 step이 포함돼 있어야 합니다.
- 과거 관측이 최소 개수보다 적으면 CV는 계산하고 CA는 `insufficient_history`로 생략합니다.
- 중간 correction이 있는 먼 query는 거절합니다. 다음 keyframe부터 검증하세요.
- mid-camera query는 이번 버전에서 지원하지 않습니다. 카메라별 시각 및 보간 GT의
  평가 정의가 필요한 별도 확장입니다.

## 테스트

```bash
python -m unittest discover -s ctmot/proto/tests -v
```

기존 ODE toy test와 함께, 실제 assignment 기반 history 선택, CTRV 속도 변환,
불규칙 시간 간격의 가속도 회귀, 미래 관측/GT 변경에 대한 prediction 불변성,
동일 초기 상태 및 Z 유지, 부족한 history 처리, 중간 correction/누락 step 거절,
원본 출력 검증, config 검증을 검사합니다.
