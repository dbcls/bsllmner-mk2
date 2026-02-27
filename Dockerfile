FROM python:3.10.18-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

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
RUN uv sync --frozen --all-extras

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["sleep", "infinity"]
