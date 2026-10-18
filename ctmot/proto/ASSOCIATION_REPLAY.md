# CT prediction association replay

이 runner는 `scene-1072 / ID102 / frame19→20`의 예측 개선이 실제 할당과 이후 ID 연속성으로 이어지는지 검증합니다. 프레임 번호는 0부터 시작합니다.

```bash
conda activate fusionpoly-repro
python -m ctmot.proto.association_replay \
  --config ctmot/proto/config/replay_scene1072.yaml
python -m unittest discover -s ctmot/proto/tests -v
```

## 실행 방식

1. 원본 high 설정, detection 및 nuScenes metadata로 scene 시작부터 frame 19까지 원래 tracker를 실행합니다. `results.json`만으로 복원할 수 없는 KF 공분산, score history, lifecycle, ID seed를 확보합니다.
2. frame 19 posterior에서 tracker를 독립적으로 복제합니다. 모든 분기는 같은 중간 camera 입력을 처리합니다.
3. frame 20에서 ID102에 한 번만 prediction을 주입합니다. mean을 KF 내부 state, FrameObject, box, projection/geometry cache에 함께 반영합니다. 그림이나 비용행렬만 바꾸는 실험이 아닙니다.
4. 기존 `Tracker.data_association()` → `matching_cost()` → `tras_update()`를 실행합니다. mix → pure3D → 2D matching cascade와 Hungarian, gate, noise/score/lifecycle 설정을 그대로 사용합니다.
5. frame 21·22 및 사이 camera 단계는 분기별 posterior에서 원래 tracker를 계속 실행합니다. CT 재주입이나 baseline state로의 복원은 없습니다.
6. 추적 완료 후 GT를 읽어 올바른 detection과 오차를 평가합니다. GT 및 query detection을 predictor의 입력으로 사용하지 않습니다.

## Variant의 정확한 의미

| Variant | 시작 위치 / 속도 | 예측 시간 | frame 20 주입 |
|---|---|---|---|
| baseline | KF posterior / 내부 CTRV | 0.25초 × 2 | 없음 |
| actual_dt_ctrv | 같은 posterior / 같은 CTRV | 0.549659초 | CTRV 평균 9차원, yaw 포함 |
| ct_cv | 같은 posterior / speed·yaw에서 변환한 XY 속도, vz=0 | 0.549659초 | CT의 XYZ 중심만 |
| ct_ca | 같은 posterior·초기 속도 / 과거 accepted detection velocity의 OLS 가속도 | 0.549659초 | CT의 XYZ 중심만 |
| past_xyz_cv | frame 19 aligned detection / frame 18·19 검출 위치 XYZ 차분 속도 | 0.549659초 | XYZ 중심만 |

모든 variant에서 원래 predicted covariance를 유지합니다. CV/CA의 Cartesian 속도를 CTRV에 전부 주입하면 내부 speed와 yaw가 바뀌면서 박스 방향까지 달라집니다. 기존 `1.0495 / 1.0200` 비교 조건을 유지하기 위해 현재 실험은 **CT 위치 평균 개입**으로 제한합니다. CT가 계산한 속도는 JSON에 별도로 기록하며, 원래 CTRV 속도·yaw·yaw-rate를 그대로 둔 채 기존 KF가 correction합니다. 따라서 완전한 CT state/covariance 필터 교체 실험은 아닙니다.

`actual_dt_ctrv`도 actual Δt로 평균을 바꾼 control이며, covariance의 시간 전파까지 바꾼 완전한 B1 EKF는 아닙니다.

## 과거 0.3182와 현재 1.0495의 차이

좌표는 nuScenes global frame이며 단위는 m, m/s입니다.

| 항목 | Past-only XYZ CV | 현재 CT-CV |
|---|---|---|
| t0 | 1542799711898267 μs | 동일 |
| tq | 1542799712447926 μs | 동일 |
| 초기 위치 | (681.045242, 2134.318649, 1.666949) | (680.826711, 2134.381927, 2.006793) |
| 초기 속도 | (6.743591, 9.529100, -0.951018) | (5.859264, 8.566563, 0) |
| 속도 추정 | 두 accepted aligned detection 위치 차분 / 0.450110초 | posterior의 speed·yaw를 Cartesian XY로 변환 |
| 예측 중심 | (684.751918, 2139.556404, 1.144214) | (684.047308, 2139.090615, 2.006793) |
| 비교 detection | 같은 frame 20 mix index 0, GAAM-aligned | 동일 |
| size / yaw | frame 20 원본 prediction 값 | 동일 |
| cost | 동일한 원본 `1-fast_giou_3d` → 0.318176 | 1.049496 |

올바른 detection의 Z는 1.157999m입니다. Past XYZ CV는 예측 Z가 1.144214m로 근접하고 현재 CT-CV는 2.006793m로 고정되어 있습니다. 초기 XY 및 XY 속도도 다르므로 비용 차이를 Z 하나 또는 연속시간 모델 자체의 효과로 귀속할 수 없습니다.

원본은 keyframe19→mid→keyframe20에서 두 번 각각 0.25초를 사용합니다. 실제 keyframe 간격은 0.549659초입니다. ID102는 그 중간 camera 단계에서 2D association되지 않았으며 motion correction도 없습니다. 중간 6개 camera의 실제 촬영 시간은 서로 다르므로 통합 mid 시간을 실제 timestamp로 해석하지 않습니다. 각 시간은 `report.json/timing`에 기록합니다.

## 결과 및 검증

기본 출력: `ctmot/proto/outputs/association_replay/scene-1072_track102_frames18-22/`

- `initial_conditions.json`, `initial_conditions.csv`: 모든 variant의 초기 state·시간·속도 source·geometry·주입 정책.
- `comparison.csv`: frame 18–22의 예측 오차, raw cost, 실제 Hungarian matrix cost, 선택 detection, correct detection owner ID, lifecycle, 출력 여부.
- `replay_records.json`: key/mid의 전체 track predictions, 모든 3D cost matrices 및 assignment, 2D match, birth lineage, posterior, 출력.
- `report.json`: baseline 일치 검사, input/code hash, timing, 다른 객체 영향, 결과 요약.
- `report.md`, `replay_comparison.png`, `prediction_centers.png`: 보고서 및 그림.

원본 high 출력과 frame 0–22의 ID 및 수치가 정확히 일치해야 계속 진행합니다. 저장된 trace와 중첩되는 모든 key/mid의 cost matrix 및 assignment도 비교합니다. frame 20에서는 ID102 외의 prediction과 first-stage cost column이 변하지 않았는지 검증합니다. 원본 입력 파일은 실행 전후 hash가 같아야 합니다.

기본 scene 결과는 baseline의 correct detection owner가 frame 18–22에서 `102, 102, 124, 124, 124`이고, 네 개 intervention에서는 모두 `102, 102, 102, 102, 102`입니다. correct detection에 대해 공간 gate를 통과한 다른 track은 없었습니다. 이 구간은 경쟁 track에 빼앗긴 사례보다 **정상 detection의 비용 탈락과 신규 ID 생성**으로 설명됩니다.

다른 객체는 ID 번호뿐 아니라 `(birth sample token, stage, detection index)` lineage로 비교합니다. ID124 생성이 줄어들면 이후 무관한 객체의 숫자 ID가 한 칸씩 달라질 수 있습니다. 이것을 새로운 association failure로 집계하지 않습니다.

여기서 ID continuity는 해당 GT에 대응하는 검출을 계속 ID102가 받는지의 검사입니다. 공식 nuScenes IDS/AMOTA 재평가는 실행하지 않습니다. 전체 val 성능이나 scene-0099 개선까지 주장하지 않습니다.

## 다른 case

replay YAML의 `validation_config`를 다른 validation YAML로 연결하고 window를 조정합니다. validation YAML에서 scene/track/anchor/query/GT ID를 지정합니다. 현재 adapter는 val car CTRV, geometry filter 미사용, 다음 keyframe에서의 단일 개입을 지원합니다. 해당 scene의 원본 결과와 정확하게 일치하는 archived trace가 필요하며, 실제 motion correction이 anchor와 query 사이에 있었다면 같은 초기 조건 비교를 위해 실행을 거부합니다.
