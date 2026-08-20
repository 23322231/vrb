# =============================================================================
# VRB (affordance) + GroundingDINO + LangSAM + Robot Arm (ROS2 / Conda)
# Base: CUDA 12.8.1 + cuDNN, Ubuntu 22.04
#
# 重要說明：
# GroundingDINO 與 LangSAM 是以「host bind mount」+「pip install -e」的方式安裝，
# 因為 /workspaces/GroundingDINO、/workspaces/LangSAM 在 build image 時根本不存在
# （它們是 container 啟動時才被掛載進來）。
# 因此這兩個套件的 editable install 不寫在這裡，而是寫在 devcontainer.json 的
# postCreateCommand，確保每次 container 建立完成、掛載完畢後自動安裝。
#
# 目前 Miniforge / Conda 整段停用（第 6 節），因為手臂的
# environment2~4.yaml 尚未備妥。第 8 節與此相關的指令已改成
# 「檢查檔案/指令是否存在才執行」，因此即使 Conda 不存在，
# build 也不會失敗；未來重新啟用 Miniforge 時不需要再改這裡。
# =============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    TZ=Asia/Taipei \
    QT_X11_NO_MITSHM=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    ROS_DISTRO=humble \
    ROS_PACKAGE=desktop-full \
    EGL_PLATFORM=surfaceless \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

# -----------------------------------------------------------------------------
# 1. 系統核心工具與環境設定
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales locales-all tzdata sudo bash-completion \
    build-essential software-properties-common gdb cmake git curl wget vim nano \
    manpages-dev gnupg2 lsb-release autoconf automake libtool bsdmainutils \
    dirmngr gosu iputils-ping less libssl-dev pkg-config \
    ca-certificates sed unzip tree zip openssh-client \
    ffmpeg \
    && locale-gen en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. 圖形介面、Librealsense 依賴（手臂視覺功能需要；音效相關已移除）
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-wqy-zenhei fonts-noto-cjk \
    libprotobuf-dev protobuf-compiler zlib1g-dev swig valgrind \
    openssl libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 \
    libgl1-mesa-dev libjpeg-dev libpng-dev language-pack-en libgomp1 \
    libboost-all-dev libasio-dev libvulkan1 mesa-utils \
    x11-apps libxcb-cursor0 xauth x11-utils xserver-xorg-core \
    libgl1-mesa-glx libgl1-mesa-dri libxcb1 libxcb-glx0 libxcb-dri2-0 \
    libxcb-dri3-0 libxcb-present0 libxcb-sync1 libxshmfence1 libxxf86vm1 \
    libsm6 libxrender1 libglib2.0-0 \
    && echo "deb [trusted=yes] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
    tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    librealsense2-utils librealsense2-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 3. 系統 Python（給 ROS2 / 手臂主程式使用，鎖定 NumPy 1.x 避免 ROS2 衝突）
#    此區塊安裝到系統 Python，與下方 VRB 專用的 /opt/venv 完全隔離
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-serial python-is-python3 python3-pip \
    python3-pytest python3-setuptools python3-dev python3-venv python3-argcomplete \
    && python3 -m pip install --upgrade pip && pip install uv \
    && python3 -m uv pip install --system \
        "numpy==1.23.5" \
        "setuptools==69.5.1" \
        scipy matplotlib cython pytest \
        pyserial pymodbus pyads \
        "ultralytics>=8.0.0" "opencv-python==4.9.0.80" \
        "torch==2.1.2" "torchvision==0.16.2" "Pillow>=10.0.0" \
        transforms3d \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 4. ROS2 humble 與機器人相關套件
# -----------------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep python3-vcstool python3-colcon-common-extensions \
    ros-${ROS_DISTRO}-${ROS_PACKAGE} ros-dev-tools \
    ros-${ROS_DISTRO}-joint-state-publisher ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-ros2-control ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-moveit ros-${ROS_DISTRO}-usb-cam \
    ros-${ROS_DISTRO}-librealsense2* ros-${ROS_DISTRO}-realsense2-* \
    ros-${ROS_DISTRO}-rtabmap-ros ros-${ROS_DISTRO}-py-trees* \
    ros-${ROS_DISTRO}-tf-transformations \
    && rosdep init || echo "rosdep already initialized" && rosdep update \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 5. 特殊機器人控制套件
# -----------------------------------------------------------------------------
RUN python3 -m uv pip install --system "pyRobotiqGripper[all]"

# -----------------------------------------------------------------------------
# 6. Miniforge / Conda（目前整段停用）
#
#     等你準備好 environment2~4.yaml，把下面整段的 "#" 拿掉即可重新啟用，
#     不需要修改第 8 節的使用者設定（那邊已經改成自動偵測 Conda 是否存在）。
#
# RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
#     bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/miniforge3 && \
#     rm Miniforge3-$(uname)-$(uname -m).sh
#
# COPY environment2.yaml /tmp/environment2.yaml
# RUN /opt/miniforge3/bin/conda env create -f /tmp/environment2.yaml && \
#     /opt/miniforge3/bin/conda clean -afy
#
# COPY environment3.yaml /tmp/environment3.yaml
# RUN /opt/miniforge3/bin/conda env create -f /tmp/environment3.yaml && \
#     /opt/miniforge3/bin/conda clean -afy
#
# COPY environment4.yaml /tmp/environment4.yaml
# RUN /opt/miniforge3/bin/conda env create -f /tmp/environment4.yaml && \
#     /opt/miniforge3/bin/conda clean -afy
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 7. VRB 專用 Python virtual environment
#    只包含「不依賴 host mount」的部分：torch/torchvision (cu128) 與基礎套件，
#    以及可直接從 Git 安裝、與 mount 無關的 segment_anything。
#    GroundingDINO 與 LangSAM 的 editable install 移到 postCreateCommand。
# -----------------------------------------------------------------------------
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

RUN /opt/venv/bin/pip install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN /opt/venv/bin/pip install \
    einops==0.6.1 \
    matplotlib==3.7.1 \
    numpy==1.24.2 \
    opencv-python==4.7.0.72 \
    scikit-learn==1.2.2 \
    "poetry-core>=1.4.0"

# segment_anything 不依賴 host mount，可直接在 image 內先裝好
RUN /opt/venv/bin/pip install \
    git+https://github.com/facebookresearch/segment-anything.git

# huggingface-hub / transformers 明確固定版本，避免之後 GroundingDINO
# editable install 時自動把它們升級掉（實測 VRB 需要這兩個精確版本）
RUN /opt/venv/bin/pip install \
    huggingface-hub==0.13.4 \
    transformers==4.27.4

# -----------------------------------------------------------------------------
# 8. 使用者建立：23322231（取代原本手臂範例中的 ncku_csie_rl）
#
#    以下與 Conda 相關的兩處，都已改成「先確認存在才執行」：
#    - 不再直接呼叫 /opt/miniforge3/bin/conda（若不存在會讓 build 失敗）
#    - .bashrc 裡的 conda.sh 來源改用 if [ -f ... ]，比照 ROS2 那一行的寫法
# -----------------------------------------------------------------------------
ENV USERNAME=c23322231 \
    USER_UID=1000 \
    USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    usermod -aG audio $USERNAME && \
    chown -R $USERNAME:$USERNAME /opt/venv

RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /home/$USERNAME/.bashrc && \
    echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc && \
    echo "if [ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash ]; then source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash; fi" >> /home/$USERNAME/.bashrc && \
    echo "if [ -f /opt/miniforge3/etc/profile.d/conda.sh ]; then source /opt/miniforge3/etc/profile.d/conda.sh; fi" >> /home/$USERNAME/.bashrc && \
    echo 'if [ -f /workspaces/NCKU_CSIE_RL_Robot/install/setup.bash ]; then source /workspaces/NCKU_CSIE_RL_Robot/install/setup.bash; fi' >> /home/$USERNAME/.bashrc && \
    if [ -x /opt/miniforge3/bin/conda ]; then /opt/miniforge3/bin/conda config --set auto_activate_base false; fi && \
    chown $USERNAME:$USERNAME /home/$USERNAME/.bashrc && \
    echo "source /home/$USERNAME/.bashrc" >> /home/$USERNAME/.bash_profile && \
    chown $USERNAME:$USERNAME /home/$USERNAME/.bash_profile

# -----------------------------------------------------------------------------
# 9. PATH 設定：VRB venv 最優先 > 系統 Python/ROS2 > Conda
# -----------------------------------------------------------------------------
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH:/opt/miniforge3/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    AMENT_CPPCHECK_ALLOW_SLOW_VERSIONS=1 \
    QT_QPA_PLATFORM=xcb \
    NO_AT_BRIDGE=1 \
    LIBGL_ALWAYS_INDIRECT=0 \
    MESA_GL_VERSION_OVERRIDE=3.3 \
    LIBGL_ALWAYS_SOFTWARE=0 \
    XDG_RUNTIME_DIR=/tmp/runtime-c23322231

# 註：不設定 WORKDIR / USER，改由 devcontainer.json 的
# workspaceFolder 與 remoteUser 統一決定。

CMD ["/bin/bash"]