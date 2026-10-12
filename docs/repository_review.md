# Fusion-Poly 연구 baseline 검토

검토일: 2026-09-19. 대상 workspace: `/workspace/Fusion-Poly`.
`.git/HEAD`와 branch ref 기준 branch는 `test`, commit은
`ca78fb652ca8be4c7df7f47bef14e3866c254d71`이다.
환경에 git 실행 파일이 없어 working tree와 commit의 차이는 확인하지 못했다.
사용자 요청에 따라 기존 val 결과 분석과 전체 val 재실행은 생략했다.
이 문서 외에 소스와 실험 설정은 변경하지 않았다.

## 판단

공간 융합, 단계별 association, 비동기 관측, confidence/lifecycle을 분리해
연구하기 좋은 tracking-by-detection baseline 구조다. 그러나 현재 checkout은
일반 실행을 막는 누락 import와 입력 변환 오류가 있어 재현 완료 baseline으로
간주할 수 없다. README의 수치는 재현 목표이며 이번 검토에서 검증한 성능이 아니다.

## 실행 및 모델 구조

| 구성 | 역할 / 주의점 |
| --- | --- |
| `test.py` | inference entry point, 결과 저장, nuScenes tracking 및 detection 평가. 학습 테스트 파일이 아니다. |
| `dataloader/nusc_loader.py` | 2D JSON 순서대로 프레임 순회. 3D projection과 2D association, coordinator, 3D-only score filtering/NMS. |
| `pre_processing/data_fusion.py` | 클래스별 2D overlap cost와 assignment로 cross-modal matching. |
| `geometry/nusc_location_coordinator.py` | projection 기반 observation 보정. 기본 `size_prior_weight=1e4`는 xyz만 최적화하는 경로를 선택한다. |
| `tracking/nusc_tracker.py` | prediction → fused observation matching → remaining 3D matching → remaining 2D matching → update/birth/death/output. |
| `motion_module/kalman_filter.py` | 클래스별 motion model과 Kalman filter. 비동기 2D update는 기본 gamma=1e12로 관측 영향을 크게 낮춘다. |
| `tracking/nusc_score_manage.py`, `nusc_life_manage.py` | sync/async score update, 평균 score 기반 종료, 출력 가능한 미관측 기간 관리. |
| `utils/inspection.py` | 저장된 클래스별 best-MOTA threshold에서 공식 평가 입력을 구성하고 IDS 이벤트/검토 구간 추출. |
| `utils/viz.py`, `utils/viz_3D.py` | RGB 및 interactive 3D 시각화. 평가 matching 자체는 수행하지 않는다. |

low/high YAML은 `basic.freq`와 `LiDAR_interval`만 다르다. low는 0.5초/2 Hz,
high는 0.25초/~4 Hz이고, 둘 다 최종 결과에는 keyframe만 저장한다.
high의 중간 프레임은 기존 track을 유지·갱신하지만 2D만으로 새로운 3D track을 만들지는 않는다.

README 기준 val 목표는 CenterPoint + Cascade R-CNN으로 low AMOTA 76.9%, IDS 446,
high AMOTA 77.1%, IDS 416이다. test의 76.5%는 Mv2DFusion 입력이므로 val 조건과
그대로 비교해서는 안 된다. 논문의 비동기 관측 활용 취지는
[arXiv v2](https://arxiv.org/abs/2603.08199v2)에서도 확인했다.
논문 전체의 수식·표와 구현이 일치하는지에 대한 전수 검증은 수행하지 않았다.

## 확인된 결함

### 1. 일반 inference도 누락된 auto-finetune 모듈을 요구한다

`test.py:6`에서 `auto_finetune_module.utils.config_dict_tree`를 무조건 import한다.
해당 디렉터리는 없고 `.gitignore`에서도 제외되어 있다.
`PYTHONDONTWRITEBYTECODE=1 python test.py --help`로
`ModuleNotFoundError: No module named 'auto_finetune_module'`를 재현했다.
auto-finetune을 쓰지 않는 실행에서는 이 import가 필요하지 않도록 분리할 필요가 있다.

### 2. 2D 재생성 경로의 세 클래스가 tracker와 불일치한다

`data/script/concat_2d_detection.py:21`의 `CLASS_TRANSFER`를
`data/script/NUSC_CONSTANT.py:23`의 tracker class mapping과 비교했다.

| 원래 detector class | 변환 번호 | tracker가 해석하는 class | 필요한 번호 |
| --- | --- | --- | --- |
| motorcycle | 0 | bicycle | 3 |
| pedestrian | 3 | motorcycle | 4 |
| bicycle | 4 | pedestrian | 0 |

car/bus/trailer/truck은 일치한다. 이 스크립트로 만든 중간 프레임 입력은 세 클래스의
정상적인 association을 막거나 잘못된 class끼리 연결할 수 있다. 기존 배포 JSON이
동일한 문제를 갖는지는 해당 파일을 확인하지 않아 알 수 없다. car만 시각화하면
이 문제를 놓칠 수 있다. mapping 수정 후에는 기존 입력과 수정 후 생성 입력을 구분해야 한다.

### 3. 3D detection이 없는 keyframe에서 loader가 중단된다

`dataloader/nusc_loader.py:156`의 nonempty branch에서만 `np_dets_insec`를
정의하고 `:245`에서 항상 참조한다. 3D 결과가 빈 리스트인 합성 프레임으로
`UnboundLocalError`를 재현했다. projection metadata 함수만 stub으로 대체했다.
또한 이 branch 밖에서는 해당 keyframe의 2D-only 관측을 채우지 않아,
변수 초기화뿐 아니라 3D 부재 시 2D 관측 보존도 검토해야 한다.
입력에 지원 tracking class만 남겼을 때 빈 프레임이 되는 경우도 같은 경로다.

### 4. multiprocess 결과 수집이 실패를 명확히 전파하지 않는다

`test.py:99`는 `result_path + str(token) + '.json'`에 저장하고,
`:461`은 `os.path.join(result_path, str(token) + '.json')`에서 읽는다.
result_path의 끝에 `/`가 없으면 서로 다른 파일을 가리킨다.
`:454`의 `apply_async` 반환값을 수집하지 않아 worker 예외가 부모에 재발생하지 않고,
merge 예외도 출력만 하고 부분 결과를 저장한다. 공식 평가에서 누락을 잡더라도
원래 원인 파악이 어려워진다. 이번 항목은 정적 코드 검토이며 multiprocess 실행은 하지 않았다.

## baseline을 정의할 때 고정할 조건

- 논문 버전, source revision와 수정 사항, detector 파일의 출처/hash, low/high 설정,
  dataset/devkit 버전, 결과와 평가 config를 함께 보관한다.
- `test.py`의 tracking/detection 평가 함수는 `val`/`v1.0-trainval`로 고정돼 있다.
  loader의 `basic.split=test` 지원과 전체 entry point의 test 지원을 혼동하면 안 된다.
- loader는 2D JSON key 순서를 그대로 사용하고 실측 시간 대신 frame index를 전달한다.
  `_mid`는 concat 코드상 **이름에 포함된 keyframe 직전**에 놓인다.
  scene 연속성, first-token 경계, keyframe당 중간 프레임 수와 실제 timestamps를 검증해야 한다.
  high는 0.25초 고정 dt이므로 불규칙 입력에도 자동으로 적응하는 구조는 아니다.
- covariance 예측은 Kalman filter의 `is_key_frame` branch에서만 수행한다.
  비동기 state prediction은 수행하므로, uncertainty 연구에서는 이 차이를 별도 검토해야 한다.
- `association.stage`만 바꾸면 해당 association 단계를 제한하지만
  `tras_update`는 여전히 3D-only observation으로 새 track을 생성한다.
  `ablation.asso` 역시 앞단 fusion/coordinator 전체를 제거하는 스위치가 아니다.
  ablation 이름만으로 실제 제거한 기능을 판단하지 말고 실행 경로를 확인해야 한다.
- 기본 coordinator는 `multi_view=false`, xyz 보정만 활성화돼 있다.
  size 보정이나 multi-view 실험은 기본 baseline과 구분해 기록한다.
- `TIME_COST_ROOT`가 고정돼 있고 각 frame에서 timing JSON을 다시 쓴다.
  병렬 실험은 timing 파일을 덮어쓸 수 있으므로 현재 로그를 정밀 속도 비교에 쓰기 어렵다.
- auto-finetune은 val 지표로 parameter를 선택한다. 향후 개선 실험에서는 tuning에
  사용한 scene과 효과 확인에 사용하는 scene을 구분하고 full-val 수치는 별도로 보고한다.

## failure case 분석 시 이어갈 흐름

`utils/inspection.py`는 설치된 nuScenes-devkit 평가 코드와 같은 best-MOTA threshold
선택, 거리 gating, evaluator의 track 평균 score 및 interpolation 경로를 사용한다.
클래스별 IDS/FP/FN/TP 합계를 저장된 summary와 대조하고 results SHA256도 기록한다.
합성 시퀀스 `GT→track a`, 한 프레임 미관측, `GT→track b`에서는 MATCH=1,
MISS=1, SWITCH=1과 이전 매칭 frame 0/현재 frame 2를 정확히 추출했다.
실제 val 결과에서 validation PASS인지 확인하는 절차는 아직 남아 있다.

현재 도구는 FP/FN을 검증용으로 세지만 개별 사건으로 내보내는 것은 IDS뿐이다.
AMOTA 전체 threshold sweep의 failure 분포를 추출하는 도구도 아니다.

나중에 결과를 생성하면 다음 순서로 분석을 이어가는 것이 좋다.

1. 같은 실행의 results와 tracking metrics를 inspection에 넣고 validation PASS 확인.
2. IDS 이벤트의 scene/class/GT instance/이전 ID/현재 ID/전후 frame으로 사례 선택.
3. raw 3D·2D → fusion/coordinator → association stage → score/lifecycle → output을
   비교해 detector 누락, fusion 오연결, association 실패, track 종료/재생성을 구분.
4. low/high를 같은 GT instance와 시간 구간에서 비교해 개선된 사례와 악화된 사례를 함께 기록.
5. 이후 필요하면 FP/FN/fragmentation과 거리·가림·회전·카메라 경계 조건으로 분석을 확장.

RGB config는 현재 `scene-0915`, car, tracking overlay=false이고,
3D config는 `scene-0099`, car, score threshold=0.2다.
같은 사례를 비교하려면 scene/frame/class/overlay 조건을 맞춰야 한다.
두 viewer는 raw prediction을 그리므로 공식 평가의 track 평균 score, interpolation,
거리/class-range filtering을 거친 box와 다를 수 있다. 화면에서 box가 없다는 이유만으로
공식 FN으로 분류하거나, ID 색상이 바뀌었다는 이유만으로 공식 IDS로 단정하면 안 된다.
3D viewer는 keyframe만 표시하고 RGB 도구는 중간 프레임의 2D 관측도 표시할 수 있다.

## 이번 검증 범위

- Python 3.7.17; `pip check` 통과. 설치 의존성 호환성 검사이며 완전한 실행 검증은 아니다.
- Python 파일 41개 AST parsing 통과.
- scene 목록은 설치된 devkit의 공식 train 700개 / val 150개와 정확히 일치.
- 누락 import, 클래스 mapping 불일치, 빈 3D frame 오류 확인.
- IDS 추출기 합성 시퀀스 확인. 실제 전체 val 평가 및 GUI 실행은 수행하지 않음.
- 현재 workspace에 detection 3D/2D, first-token table, `Fusion_Poly_EXP`가 없다.
  `/data/nuScenes`에는 trainval blob archive 한 개가 보이고 필요한 metadata 디렉터리는 없다.
  데이터 준비와 기존 결과 확보는 사용자 요청에 따라 이번 범위에서 제외했다.
