# macOS Docker + XQuartz 초안

현재 tracker의 Python 3.7 / Open3D 0.17 환경을 Linux 컨테이너로 분리하고,
소스와 데이터는 맥에서 bind mount한다. Apple Silicon도 기존 x86 의존성 조합을
유지하도록 `linux/amd64`로 실행한다. 따라서 Apple Silicon에서는 에뮬레이션 비용이 있다.
이미지에는 소스·dataset·detection·tracking 결과를 복사하지 않는다.

| Mac host | Container | 용도 |
|---|---|---|
| `~/allen_workspace/git/baseline_3DMOT/Fusion-Poly` | `/workspace/Fusion-Poly` | 소스·결과, 읽기/쓰기 |
| `~/dataset/nuScenes` | `/data/nuScenes` | nuScenes 원본, 읽기 전용 |
| `~/.cache/fusionpoly/Xauthority` | `/tmp/fusionpoly.Xauthority` | XQuartz 인증, 읽기 전용 |

dataset root는 `samples/`, `sweeps/`, `v1.0-trainval/`을 직접 포함한다고 가정한다.
실제로 `full_trainval/` 같은 하위 폴더가 있다면 local config의 dataset_root를
`/data/nuScenes/full_trainval`로 지정한다. Git에서 제외한 detection inputs 및
`Fusion_Poly_EXP/result/nusc_config_high/results.json`도 맥 workspace에 별도로 준비해야 한다.
Linux의 절대 경로를 가리키는 symlink는 이 컨테이너에서 자동으로 연결되지 않는다.

## GUI 범위

이 구성은 **컨테이너 내부 Mesa/llvmpipe CPU 렌더링 + XQuartz 화면 출력**을 시도한다.
Mac GPU/Metal 하드웨어 가속이나 CUDA passthrough를 설정한 구성이 아니다.
`gui.require_hardware_gpu: false`는 local config에만 적용한다.

Open3D 0.17의 `visualization.gui`는 OpenGL 4.1 이상이 필요하다.
X11 연결 성공이나 `xeyes` 실행 성공만으로 Open3D 실행을 보장할 수 없다.
실제 Mac의 XQuartz/GLX visual/context 지원 여부는 아래 순서로 확인해야 한다.
`LIBGL_ALWAYS_INDIRECT=1`, `MESA_GL_VERSION_OVERRIDE`로 요구사항을 우회하지 않는다.
소프트웨어 렌더링도 GL context 생성에 실패한다면 해당 환경에서는 이 초안의
Open3D GUI 경로를 사용할 수 없다. 그 경우 Mac native Open3D 또는 Linux GPU
desktop의 원격 화면 방식으로 렌더링 경로를 바꿔야 한다. 데이터 검증/RGB 파일 렌더링은
Open3D GUI context를 필요로 하지 않는다.

참조: [Open3D 0.17 CPU rendering](https://www.open3d.org/docs/0.17.0/tutorial/visualization/cpu_rendering.html),
[Docker Desktop networking](https://docs.docker.com/desktop/features/networking/),
[XQuartz FAQ](https://www.xquartz.org/FAQs.html).

## 1. Mac에서 XQuartz 준비

Docker Desktop과 XQuartz가 설치되어 있어야 한다. XQuartz Settings/Preferences →
Security에서 **Allow connections from network clients**를 켜고 **Authenticate connections**는
유지한다. XQuartz를 완전히 종료한 뒤 다시 실행한다. 신규 설치 후 DISPLAY가 설정되지
않으면 로그아웃/로그인하거나 XQuartz의 Applications → Terminal에서 아래를 실행한다.

```bash
cd "$HOME/allen_workspace/git/baseline_3DMOT/Fusion-Poly"
bash docker/xquartz-auth.sh
```

인증 cookie만 별도 파일로 export한다. `xhost +`는 필요하지 않다.
XQuartz를 재시작하면 이 명령을 다시 실행하고 컨테이너도 새로 시작한다.
기본 display 번호 `:0`을 가정한다. 다른 번호라면 compose.yaml의 DISPLAY도 변경한다.
macOS 방화벽에서 Docker Desktop의 XQuartz 연결이 허용되어야 한다.

## 2. 빌드 및 X11/OpenGL 점검

```bash
docker compose build fusionpoly
docker compose run --rm fusionpoly xeyes
# xeyes 창을 닫은 뒤 실행
docker compose run --rm fusionpoly glxinfo -B
```

`glxinfo -B`에서 `llvmpipe` renderer와 **OpenGL core profile 4.1 이상**을 확인한다.
`unable to open display` / authorization 오류면 XQuartz TCP 설정 및 cookie를 확인한다.
`GLXBadFBConfig`, context 생성 오류 또는 OpenGL 2.1만 보이면 Open3D 지원 확인이
실패한 것이다. 2D 창이 떠도 그 상태에서 3D 실행 성공으로 판단하지 않는다.

GUI 코드를 실행하기 전 데이터 없이 작은 Open3D 창을 열어볼 수도 있다:

```bash
docker compose run --rm fusionpoly python -c \
  'import open3d as o3d; from open3d.visualization import gui; a=gui.Application.instance; a.initialize(); w=a.create_window("Open3D XQuartz check",640,480); s=gui.SceneWidget(); s.scene=o3d.visualization.rendering.Open3DScene(w.renderer); w.add_child(s); a.run()'
```

## 3. 컨테이너용 local 설정 생성

공유 설정을 읽어 `config/viz_rgb.local.yaml`, `config/viz_3D.local.yaml`을 만든다.
기존 local 파일은 덮어쓰지 않는다. 아래 코드는 host의 전체 설정을 복사하므로
scene/class/score 옵션은 현재 공용 YAML 값을 유지한다.

```bash
docker compose run --rm -T fusionpoly python - <<'PY'
from pathlib import Path
import yaml

for name in ('viz_rgb', 'viz_3D'):
    source = Path('config') / (name + '.yaml')
    target = Path('config') / (name + '.local.yaml')
    if target.exists():
        print('Keep existing:', target)
        continue
    cfg = yaml.safe_load(source.read_text())
    cfg['paths'].update({
        'dataset_root': '/data/nuScenes',
        'results_json': 'Fusion_Poly_EXP/result/nusc_config_high/results.json',
        'detection_3d_json': 'data/detector/val/nuscenes_val_centerpoint_3d.json',
        'detection_2d_json': 'data/utils/cascade_rcnn_4hz/nuscenes_val_cascade_2d_4hz.json',
        'output_dir': 'Fusion_Poly_EXP/visualization/docker_' + name,
    })
    cfg['selection']['scene_name'] = 'scene-0099'
    if name == 'viz_3D':
        cfg['gui']['require_hardware_gpu'] = False
    target.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print('Created:', target)
PY
```

## 4. 실행

```bash
# 데이터/좌표 변환 검증 (Open3D 창을 열지 않음)
docker compose run --rm fusionpoly \
  python utils/viz_3D.py --config config/viz_3D.local.yaml --validate-only

# 한 scene의 interactive 3D GUI
docker compose run --rm fusionpoly \
  python utils/viz_3D.py --config config/viz_3D.local.yaml

# RGB 이미지 저장
docker compose run --rm fusionpoly \
  python utils/viz.py --config config/viz_rgb.local.yaml

# 컨테이너 개발 셸
docker compose run --rm fusionpoly bash
```

소스 수정과 렌더링 결과는 bind-mounted Mac workspace에 남는다.
requirements 또는 Dockerfile을 수정하면 이미지를 다시 빌드한다.
새 tracking 실행 시에도 `--nusc_path /data/nuScenes`를 명시한다.

## 검증 상태

초안 작성 환경은 Linux이며 Mac/XQuartz 종단간 GUI 실행은 아직 검증하지 않았다.
Dockerfile에는 빌드 시 `pip check`와 주요 Python/Open3D GUI 모듈 import 검사를 넣었다.
이는 build가 실제 성공했거나 OpenGL context 생성이 검증됐다는 뜻은 아니다.
Python 3.7 base image와 패키지 저장소의 다운로드 가능 여부도 첫 빌드에서 확인해야 한다.
