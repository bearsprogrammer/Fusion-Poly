# Development image: source and nuScenes are bind-mounted at runtime.
# Build with --platform linux/amd64 (also specified by compose.yaml).
# Python 3.7 is intentional for compatibility with the existing tracker stack.
FROM python:3.7.17-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/fusionpoly-matplotlib \
    XDG_CACHE_HOME=/tmp/fusionpoly-cache

RUN test "$(dpkg --print-architecture)" = amd64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash build-essential ca-certificates \
        libegl1 libgl1 libgl1-mesa-dri libglx-mesa0 libglu1-mesa \
        libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
        libx11-6 libx11-xcb1 libxcb1 libxi6 libxrandr2 libxinerama1 libxcursor1 \
        libxkbcommon0 libfontconfig1 fonts-dejavu-core \
        mesa-utils x11-apps x11-utils xauth \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-viz3d.txt /opt/fusionpoly/
COPY docker/constraints-py37.txt /opt/fusionpoly/constraints-py37.txt

RUN python -m pip install --no-cache-dir \
        pip==24.0 setuptools==65.6.3 wheel==0.38.4 \
    && python -m pip install --no-cache-dir --prefer-binary \
        -c /opt/fusionpoly/constraints-py37.txt \
        -r /opt/fusionpoly/requirements.txt \
        -r /opt/fusionpoly/requirements-viz3d.txt \
    && python -m pip check \
    && python -c "import numpy, cv2, scipy, numba, lap, motmetrics, nuscenes, open3d; from open3d.visualization import gui, rendering; print('Imports OK:', numpy.__version__, cv2.__version__, open3d.__version__)"

WORKDIR /workspace/Fusion-Poly
CMD ["bash"]
