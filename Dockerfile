#Dockerfile
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

#ARG,ENV 를 이용하여 오류 1을 넘김.
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
SHELL ["/bin/bash", "-c"]code .


# RUN 을 이용하여 내가 원하는 Docker Container 내부에 실행 되도록 만듬.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    pkg-config \
    software-properties-common \
    ssh \
    sudo \
    unzip \
    wget
RUN rm -rf /var/lib/apt/lists/*

# Vim
RUN sudo apt-get update && sudo apt-get install -y vim

RUN echo "PS1='\[\033[0;32m\]\u@\h:\[\033[0;34m\]\w\[\033[0m\]\$ '" >> ~/.bashrc

# CUDA tool-kit 설정
RUN echo 'PATH=$PATH:/usr/local/cuda-11.3/bin' >> ~/.bashrc
RUN echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64' >> ~/.bashrc
RUN echo 'CUDA_HOME=/usr/local/cuda-11.3' >> ~/.bashrc
RUN /bin/bash -c 'source ~/.bashrc'

# Miniconda 사용하는 방법
ENV LANG=C.UTF-8
RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm /tmp/miniconda.sh
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update -y conda