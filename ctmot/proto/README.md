# CT prediction prototype — ODE toy validation

이번 단계의 범위는 비학습 ODE predictor와 toy test입니다. Fusion-Poly의
tracking, correction, covariance, association에는 연결하지 않았습니다.

## 상태와 모델

상태는 global Cartesian 좌표의 `x = [px, py, pz, vx, vy, vz]`입니다.
위치 단위는 m, 속도는 m/s, 가속도는 m/s²입니다. 속도는 벡터이고,
반환되는 `speed_xy_mps`, `speed_3d_mps`는 각각 XY/XYZ 속력입니다.

`dynamics.py`의 `rhs(time_s, state)`가 ODE의 우변입니다.

```python
# ConstantVelocity.rhs: dp/dt = v, dv/dt = 0
return np.concatenate((state[3:6], np.zeros(3)))

# ConstantAcceleration.rhs: dp/dt = v, dv/dt = a
return np.concatenate((state[3:6], self.acceleration_mps2))
```

CA의 가속도는 config에서 지정한 고정 파라미터입니다. 검출 이력이나 GT로
추정하지 않습니다. 상태에 acceleration, yaw, 크기, covariance는 포함하지 않습니다.

## 실제 적분 코드

`predictor.py`의 `ODEPredictor.predict()`는 입력 상태의 사본을 적분합니다.
Unix timestamp를 먼저 float로 바꾸지 않고 정수끼리 차감하여 작은 시간 차이를 보존합니다.

```python
dt = (t - t0) / 1_000_000.0
solution = solve_ivp(
    self.dynamics.rhs, (0.0, dt), initial, method="RK45",
    t_eval=[dt], rtol=self.rtol, atol=self.atol,
    max_step=self.max_step_s,
)
```

- timestamp 입력: 정수 microseconds, query 시각은 상태 시각 이상.
- query가 같은 시각이면 입력과 같은 상태를 반환하며 적분을 생략합니다.
- predictor의 조회는 입력 state를 갱신하지 않습니다. 같은 초기 상태에서 서로 다른
  카메라 시각을 조회해도 조회 순서의 영향을 받지 않습니다.
- ODE 내부 시간은 `t0`로부터의 상대 초입니다. 두 모델은 시간에 명시적으로
  의존하지 않는 autonomous ODE입니다.
- `max_step_s`는 수치 적분 내부 step의 상한입니다. 관측 간격이나 고정 frame dt가
  아니며, 마지막 출력 시각은 요청한 `query_timestamp_us`입니다.
- 부적절한 state, 과거 시각, 설정 오타, solver 실패는 예외로 보고합니다.
- 해석해는 **검증에만** 사용합니다. 양수 Δt의 prediction은 실제 RK45로 계산합니다.

## 실행 예제

레포 root에서 기존 환경을 사용합니다. 추가 패키지 설치는 필요 없습니다.

```bash
conda activate fusionpoly-repro

# CV: 초기 p=(0,0,0), v=(2,-1,0.5), 2.5초 후 조회
python -m ctmot.proto.run_prediction \
  --config ctmot/proto/config/prediction_cv.yaml \
  --state 0 0 0 2 -1 0.5 \
  --state-time-us 1000000 \
  --query-time-us 3500000

# CA: 같은 초기 상태, config의 a=(1,0.4,-0.2) m/s²
python -m ctmot.proto.run_prediction \
  --config ctmot/proto/config/prediction_ca.yaml \
  --state 0 0 0 2 -1 0.5 \
  --state-time-us 1000000 \
  --query-time-us 3500000
```

검증용 해석해:

```text
CV: p(t) = p0 + v0 * dt
    v(t) = v0

CA: p(t) = p0 + v0 * dt + 0.5 * a * dt²
    v(t) = v0 + a * dt
```

| 모델 | 2.5초 후 위치 (m) | 2.5초 후 속도 벡터 (m/s) | XYZ 속력 (m/s) |
|---|---|---|---|
| CV | (5, -2.5, 1.25) | (2, -1, 0.5) | 2.2912878475 |
| CA | (8.125, -1.25, 0.625) | (4.5, 0, 0) | 4.5 |

CLI는 전체 결과를 JSON으로 출력합니다. 수치 적분 때문에 0이 약 `1e-16`,
5가 `4.999999999999999`처럼 출력될 수 있습니다. 아래 테스트는 허용 오차로 비교합니다.

Python에서 직접 호출하려면:

```python
from ctmot.proto import ConstantAcceleration, ODEPredictor

predictor = ODEPredictor(ConstantAcceleration((1.0, 0.4, -0.2)))
result = predictor.predict(
    state=[0, 0, 0, 2, -1, 0.5],
    state_timestamp_us=1_000_000,
    query_timestamp_us=3_500_000,
)
print(result.to_dict())
```

## Toy test 검증

```bash
python -m unittest discover -s ctmot/proto/tests -v
```

`tests/test_prediction.py`의 15개 test method가 통과했습니다.
검증 환경은 Python 3.7.16, NumPy 1.21.6, SciPy 1.7.3, PyYAML 6.0.1입니다.

| 검증 대상 | 확인 방법 |
|---|---|
| CV/CA 운동방정식 | 독립 계산한 해석해와 state 각 성분 비교 (`atol=1e-9`, `rtol=0`) |
| 불규칙 Δt | 1μs, 3.001ms, 250ms, 549.659ms, 2.5s, 10s |
| RK45 실제 사용 | 두 모델의 양수 Δt query에서 solver 호출 및 nfev 확인 |
| 경계 조건 | 정지, a=0인 CA와 CV 일치, Δt=0, 과거 query 거절 |
| 시간 정밀도 | `t0=10**16`μs에서도 1μs 차이 보존 |
| 순서 독립성 | 입력/가속도 사본 보존, query 순서를 바꿔도 동일 결과 |
| 분할 적분 | t0→mid→t와 t0→t가 같은 상태를 생성 |
| 오류 처리 | NaN/Inf, 잘못된 state shape/설정/timestamp, 적분 실패 |
| 공개 실행 경로 | 두 YAML과 CLI 출력이 위 손계산 예제와 일치 |

불규칙 Δt 해석해 비교에서 관측한 최대 절대 성분 오차:
CV `3.20e-14`, CA `4.26e-14` (위 환경).
이는 수치 구현의 정확성 검증이며, 실제 차량 운동 예측력이나 IDS 개선의 증거는 아닙니다.
