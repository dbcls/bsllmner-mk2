FROM python:3.10.18-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG VERSION=dev

LABEL org.opencontainers.image.source="https://github.com/dbcls/bsllmner-mk2"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.licenses="MIT"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN curl -Lo docker.tgz https://download.docker.com/linux/static/stable/x86_64/docker-28.2.2.tgz && \
    tar -xzf docker.tgz && \
    mv docker/docker /usr/local/bin/docker && \
    rm -rf docker docker.tgz

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN mkdir -p bsllmner2

# Install dependencies first (cache layer).
RUN uv sync --frozen --all-extras --no-install-project && \
    chmod -R a+rwX .venv

COPY . .

# Install the project with version injected from the build arg.
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} uv sync --frozen --all-extras

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["sleep", "infinity"]
