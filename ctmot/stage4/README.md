# Stage 4: Multi-case and full nuScenes validation

원본 tracker/detection/config/results를 수정하지 않고, `ctmot`의 ODE predictor와 Stage 3 replay를 재사용합니다. 기본 환경은 Linux의 `fusionpoly-repro`이며 추가 패키지는 필요하지 않습니다.

```bash
conda activate fusionpoly-repro
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/fusionpoly-mpl

python -m unittest discover -s ctmot/proto/tests -v
python -m unittest discover -s ctmot/stage4/tests -v
python -m ctmot.stage4.run --config ctmot/stage4/config.yaml --stage all
```

단계별 실행:

```bash
python -m ctmot.stage4.run --stage multicase
python -m ctmot.stage4.run --stage small
python -m ctmot.stage4.run --stage full  # 150 scenes tracking + official eval + report
# 이미 tracking이 끝났다면:
python -m ctmot.stage4.tracking --full --finalize-only  # scene 출력에서 병합만 재개
python -m ctmot.stage4.run --stage evaluate
python -m ctmot.stage4.run --stage report
```

`--stage full`은 multi-case와 small 검증 결과가 있어야 실행됩니다. 정상 detection preprocessing cache는 입력 hash가 같을 때 재사용합니다. tracker 결과는 각 실행에서 새로 생성합니다. cache pickle은 이 runner가 직접 생성한 로컬 파일만 사용해야 합니다.

## 선언된 5개 event

| Event | Scene | Old ID | Anchor → intervention | 분류 |
|---|---|---:|---|---|
| 419 | 1072 | 102 | 19 → 20 | P1 |
| 79 | 0099 | 152 | 19 → 20 | P1 |
| 81 | 0099 | 45 | 28 → 29 | P1 |
| 84 | 0101 | 14 | 10 → 14 | P2 |
| 349 | 0915 | 108 | 8 → 12 | P2 |

모든 frame은 0-based keyframe index입니다. P2는 검출 공백 직전의 실제 corrected posterior에서 시작합니다. official IDS 시점과 실제 신규 track 생성 시점은 다를 수 있습니다. scene-1072는 공식 event frame 21이지만 개입 대상은 신규 ID가 생성되는 frame 20입니다.

Local replay는 모든 variant를 같은 posterior/KF/P/score/lifecycle에서 분기하고 query에서 한 번만 주입합니다. 이후 query+2까지 원래 tracker를 계속 실행합니다. target이 이미 사라졌다면 되살리지 않고 absent로 보고합니다. GT는 모든 분기의 tracking 완료 후 평가용 detection correspondence와 오차 계산에만 사용합니다.

## Full-run variant

| Variant | Prediction mean | 시간 | Yaw / shape | Covariance |
|---|---|---|---|---|
| B0 | Original high | 원래 0.25 s/event | 원본 | 원본 |
| B1 | 원래 class별 model | 실제 timestamp 차이 | actual-dt 원본 model | 원본 nominal schedule |
| B2 | ODE CV의 geometric center | 실제 query - posterior time | B1과 같은 정책 | B1과 같은 정책 |
| B3 | ODE CA의 geometric center | 실제 query - posterior time | B1과 같은 정책 | B1과 같은 정책 |

원래 high는 모든 class가 CTRV가 아닙니다. bus/car/pedestrian/truck은 CTRV, bicycle/motorcycle은 BICYCLE, trailer는 CV입니다. B1에서 이를 모두 CTRV로 교체하면 timestamp-only 비교가 아니므로 원래 model을 보존합니다. (2026-10-21: 문서의 class 매핑 오류 수정. 실행 코드는 원래 numeric config 매핑을 사용했으므로 저장된 tracking 결과에는 영향이 없습니다.)

Full run은 class/scene/GT/track ID를 선택하지 않습니다. predictor에는 solver/history 설정만 넘깁니다. config의 5개 event는 local replay와 small scene 선택에만 사용됩니다.

전체 val에서 각 variant는 keyframe 6,019개와 mid-camera 묶음 5,869개, 총 11,888 event를 처리합니다. High의 mid 관측을 사용하지만 공식 tracking 지표는 keyframe 출력에 대해 계산합니다.

- 원래 `Tracker.tracking()`과 association/cascade/Hungarian/gates/KF update/score/lifecycle을 호출합니다. 단지 `tras_predict()` 후 prediction mean 및 관련 box/cache를 일관되게 갱신합니다.
- CT-CV/CA는 마지막 실제 3D correction 또는 birth에서 얻은 posterior geometric position과 물리적 XY velocity가 초기 조건입니다. camera-only 또는 unmatched 단계에서는 anchor를 갱신하지 않습니다.
- CTRV velocity는 `speed*cos(yaw), speed*sin(yaw)`입니다. 기존 results의 velocity 슬롯을 Cartesian으로 오해하지 않습니다.
- BICYCLE은 gravity center와 geometric center를 변환하고, yaw 회전에 의한 offset 미분까지 포함한 geometric center velocity를 사용합니다. finite-difference test로 검증합니다.
- 원래 CV transition도 Z를 고정하고 vz를 0으로 만듭니다. 이에 맞춰 모든 class에서 vz=az=0입니다.
- CA는 최근 최대 4개의 실제 assigned 3D detection velocity에 대해 실제 timestamp로 OLS slope를 계산합니다. birth는 가속도 관측에 포함하지 않습니다. 3개 미만이면 명시적으로 CV fallback이며 횟수를 보고합니다. clipping이나 GT 기반 parameter tuning은 없습니다.
- ODE는 prototype의 `solve_ivp(RK45)`를 그대로 사용합니다. 미래 GT·미래 detection·미래 posterior를 초기화나 가속도 추정에 사용하지 않습니다.
- 기존 state의 speed/yaw/rate, yaw, shape와 covariance 방식은 유지합니다. CT의 위치만 주입하므로 이 결과는 **CT position-mean ablation**입니다. CT velocity/covariance를 가진 완전한 필터 교체 결과가 아닙니다.
- covariance의 원래 nominal F/Q 및 key/mid 갱신 스케줄을 공통 유지합니다. B1–B3는 평균 전파의 시간·위치 model 효과를 비교합니다. 후속 posterior가 달라져 covariance가 달라지는 것은 정상적인 closed-loop 효과입니다.

## Actual timestamp

Keyframe에서는 실제 sample timestamp를 사용합니다. mid의 6개 camera는 서로 다른 촬영 시각을 가지며, 기존 pipeline은 이를 하나의 association 묶음으로 처리합니다. 묶음의 가장 늦은 camera timestamp를 처리 시각으로 사용해 현재 묶음의 모든 관측이 이미 도착한 조건을 유지합니다. camera별 timestamp는 입력 cache에 보존합니다.

이 선택은 camera 각각의 시각에 개별 prediction/association하는 새로운 pipeline은 아닙니다. 원래 matching cascade를 유지하기 위한 고정 batching 정책입니다. 모든 scene에서 처리 시각의 단조 증가를 검사합니다.

## 검증과 공식 평가

- B0의 ID/class/size/score는 원본 high와 정확히 같아야 합니다. translation/rotation 허용 오차는 1e-8, velocity는 1e-7입니다. 전체 6,019개 keyframe의 실제 최대 차이는 translation 6.62e-9 m, quaternion 원소 1.76e-9, velocity 1.94e-8이며, velocity 한 항목만 1e-8을 넘었습니다. 수치 연산 차이는 `baseline_differences.json`과 `baseline_validation_full.json`에 기록합니다. 공식 지표 일치 검사는 별도로 유지합니다.
- scene-1072의 Stage 3 비용과 ID continuity를 재검증합니다.
- full 공식 평가의 prediction sample set은 val 150개 scene 전체여야 합니다.
- 공식 devkit의 `TrackingEvaluation.accumulate()`를 class별 process에서 실행하고, 원래 `TrackingEval.evaluate()`/`main()`의 threshold 선택·집계·출력을 그대로 사용합니다. devkit 소스 파일은 수정하지 않습니다.
- B0 AMOTA/AMOTP/MOTA/IDS/Frag/FP/FN 및 class별 지표가 기존 공식 저장 지표와 일치해야 합니다.
- per-scene 자체 threshold counts의 합이 공식 전체 counts와 같은지도 검사합니다.
- scene 개선/악화 비교에는 B0의 class별 best-MOTA threshold를 모든 variant에 공통 적용합니다. `FN+FP+IDS` 합이 감소/증가하면 improved/worsened로 분류합니다. 별도로 각 variant의 own-best-MOTA 결과도 제공합니다. scene별 AMOTA라는 이름을 붙이지 않습니다.

## 결과 파일

기본 root: `ctmot/proto/outputs/stage4/`

- `multicase/comparison.csv`, `summary.csv`, `report.json`: 5개 event의 prediction/cost/gating/assignment/continuity와 다른 객체 영향.
- `multicase/<event>/initial_conditions.json`, `records.json`: 초기 조건, 전체 내부 추적 기록.
- `inputs/manifest.json`: 동일 전처리 cache의 입력 hash, scene/sample 수, 전처리 시간.
- `runs/full/B*/results.json`, `config.json`, `runtime.json`: variant별 추적 출력·실험 설정·시간.
- `evaluation/B*/metrics_summary.json`, `metrics_details.json`: 공식 평가 결과.
- `evaluation/B*/scene_metrics.json`, `provenance.json`: scene 진단·평가 검증 정보.
- `report/metrics.csv`, `runtime.csv`, `ablation.csv`, `scene_changes.csv`, `README.md`, PNG 그림: 최종 비교.
- `run_manifest.json`, `commands.txt`: 원본/코드 hash, 환경, 실행 명령.

FPS는 GPU 추론을 포함하지 않습니다. 이미 준비된 detection에서 tracking한 시간 합으로 keyframe FPS/event FPS를 계산하며 전처리 포함 수치도 제공합니다. worker가 동시에 실행되므로 시간 합과 batch wall-clock을 구분합니다. 평가 시간은 tracking FPS에서 제외합니다.
측정 구간은 `tracker.advance()` 호출입니다. cache 읽기, tracker 초기화, 입력 deepcopy 및 최종 JSON 저장은 제외합니다. `end_to_end_keyframe_fps` 필드는 전처리 시간과 이 tracking 시간을 합친 값이며 시스템 전체 end-to-end 처리량은 아닙니다.
runtime의 `counters.predictions`는 새 prediction 정책의 적용 횟수입니다. 원본 경로로 즉시 반환하는 B0의 0은 예측을 하지 않았다는 뜻이 아닙니다.
병합 중단 후 `--finalize-only`로 재개한 full run은 정확한 batch wall-clock을 복원할 수 없어 null로 기록합니다. 각 scene에서 실측한 tracking 시간과 이를 합산한 FPS는 보존됩니다.

## 현재 범위와 해석

Local 5개 event는 선택된 실패 표본이므로 복구 개수 자체를 전체 개선으로 해석하지 않습니다. 특히 P2 중 scene-0915는 과거 accepted 관측 부족으로 CA→CV fallback이 발생합니다. B1 대비 B2/B3의 추가 이점 여부는 full-val 지표, class별 trade-off, 악화 scene 및 runtime을 함께 확인해야 합니다.

CV/CA ODE는 해당 모델의 해석해와 동등합니다. 이번 비교는 timestamp와 motion prior 효과를 검증하며, ODE 수치 적분 자체의 우월성 또는 inter-frame 정보 손실이 IDS의 일반적 원인임을 증명하지 않습니다. 새 sweep 관측이나 CT association cost는 추가하지 않았습니다.

아래 결과 문서는 실행 완료 후 생성됩니다:

[최종 결과 보고서](../proto/outputs/stage4/report/README.md)

## Local replay에서 관측된 결과

| Event / target | B0 cost | B1 cost / 복구 | B2 cost / 복구 | B3 cost / 복구 |
|---|---:|---|---|---|
| 419 / 1072-ID102 | 1.147357 | 1.037978 / 성공 | 1.049496 / 성공 | 1.020027 / 성공 |
| 79 / 0099-ID152 | 1.121985 | 0.818431 / 성공 | 0.807589 / 성공 | 0.719670 / 성공 |
| 81 / 0099-ID45 | 1.147709 | 1.080012 / 실패 | 1.068958 / 성공 | 0.983377 / 성공 |
| 84 / 0101-ID14 | 1.517624 | 1.517526 / 실패 | 1.498734 / 실패 | 1.177098 / 실패 |
| 349 / 0915-ID108 | 1.301563 | 1.299342 / 실패 | 1.284541 / 실패 | 1.284541 / 실패(CV fallback) |

B4 Past XYZ CV는 P1 세 개에서 성공하고 P2 두 개에서는 실패했습니다. 1072의 Stage 3 cost/continuity는 재현됐습니다. Local 구간에서 다른 객체의 lineage assignment 및 공통 lineage state/score/lifecycle/covariance 변경은 없었습니다.

- 0099-ID45: B1은 cost 1.0800으로 cost gate 안이지만 검출까지 XYZ 거리 3.2938m로 spatial gate 밖입니다. B2는 2.9795m, B3는 2.5569m로 gate 안에 들어와 실제 할당과 ID 유지에 성공합니다. B2의 gate 여유는 약 2cm로 작은 경계 사례입니다.
- 0101-ID14: CA가 GT XY error를 5.5214m에서 2.9688m로 줄였습니다. 검출까지 거리는 2.8462m로 spatial gate를 통과하지만 cost 1.1771이 1.1을 넘어 실패합니다.
- 0915-ID108: CV의 검출까지 거리는 3.9604m로 spatial gate 밖입니다. CA에 필요한 accepted history가 부족하므로 이 결과를 충분한 관측을 사용한 CA 모델의 실패로 해석하지 않습니다.
- P2 두 사례 모두 재등장 시 기존 track은 active였습니다. 이 구간의 실패를 lifecycle 종료로 설명하지 않습니다.
