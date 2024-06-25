FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
RUN apt-get update -qq && apt-get install -qqy --no-install-recommends \
    make \
    build-essential \
    git \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app_build
COPY src/ pdm.lock pyproject.toml README.md /app_build/
RUN pip install pdm
RUN pdm use $(which python)
RUN pdm sync
RUN pip install pytorch-lightning==2.0.3 nltk>=3.8.1
RUN python -m nltk.downloader averaged_perceptron_tagger && \
    python -m nltk.downloader cmudict
RUN pdm cache clear
WORKDIR /workspace