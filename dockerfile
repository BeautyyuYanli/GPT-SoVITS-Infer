FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
COPY src/ pdm.lock pyproject.toml README.md /app_build/
RUN apt-get update -qq && apt-get install -qqy --no-install-recommends \
    make \
    build-essential \
    git \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app_build
RUN pip install pdm pytorch-lightning==2.0.3 nltk>=3.8.1 && \
    pdm use $(which python) && \
    pdm sync && \
    pdm cache clear && pip cache purge
RUN python -m nltk.downloader averaged_perceptron_tagger && python -m nltk.downloader cmudict
WORKDIR /workspace