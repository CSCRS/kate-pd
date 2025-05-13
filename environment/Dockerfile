# hash:sha256:94556dd73f16cadb47e391c24d7c11f85cadcc0d9bca2e320fc278a4ede44725
FROM registry.codeocean.com/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    albumentations \
    clip \
    scikit-image \
    segmentation-models-pytorch \
    ultralytics
