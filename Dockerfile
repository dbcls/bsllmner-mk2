FROM python:3.10.18-bookworm

RUN apt update && \
    apt install -y --no-install-recommends \
    curl \
    jq && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN curl -Lo docker.tgz https://download.docker.com/linux/static/stable/x86_64/docker-28.2.2.tgz && \
    tar -xzf docker.tgz && \
    mv docker/docker /usr/local/bin/docker && \
    rm -rf docker docker.tgz

WORKDIR /app
COPY . .
RUN python3 -m pip install --no-cache-dir --progress-bar off -U pip && \
    python3 -m pip install --no-cache-dir --progress-bar off -e .[tests]

CMD ["sleep", "infinity"]
