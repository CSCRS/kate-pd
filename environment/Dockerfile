# hash:sha256:6a5d52e483b748a8b5a749e2baded8515f1057806371fbc50f21a69071a060c3
FROM registry.codeocean.com/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx=23.0.4-0ubuntu1~22.04.1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    albumentations==1.4.18 \
    clip==0.2.0 \
    datasets==3.6.0 \
    scikit-image==0.25.2 \
    segmentation-models-pytorch==0.3.4 \
    ultralytics==8.3.120
