FROM ghcr.io/astral-sh/uv:0.12 AS uv

# === build-base: uv and a C compiler ===

FROM python:3.14-slim-trixie AS build-base

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# owlready2 has no wheels, and its install compiles the C extension owlready2_optimized.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# === builder: runtime dependencies and the project, installed into /app/.venv ===

FROM build-base AS builder

ARG VERSION=0.0.0.dev0

ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock README.md ./

# Install dependencies first (cache layer).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY bsllmner2 ./bsllmner2

# Install the project with version injected from the build arg.
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} uv sync --frozen --no-dev --no-editable

# owlready2 falls back to pure Python without an error if the C extension fails to build.
RUN .venv/bin/python -c "import owlready2_optimized"

# owlready2 bundles the jars of the Pellet reasoner. bsllmner2 does not run OWL reasoners and the
# runtime image has no Java, so the jars are never used. owlready2 lists the directory on import,
# so keep the directory and delete only the jars.
RUN rm -f .venv/lib/python*/site-packages/owlready2/pellet/*.jar

# === dev: used by compose; the repository is bind-mounted at /app ===

# dev keeps the C compiler of build-base, so that `uv sync` in the container also builds owlready2_optimized.
FROM build-base AS dev

ARG VERSION=0.0.0.dev0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
RUN mkdir -p bsllmner2

# Install dependencies first (cache layer).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project && \
    chmod -R a+rwX .venv

COPY . .

# Install the project as editable, so that the bind-mounted source takes effect without a rebuild.
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} uv sync --frozen

RUN .venv/bin/python -c "import owlready2_optimized"

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["sleep", "infinity"]

# === runtime: the published image ===

FROM python:3.14-slim-trixie AS runtime

ARG VERSION=0.0.0.dev0

LABEL org.opencontainers.image.source="https://github.com/dbcls/bsllmner-mk2"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.licenses="MIT"

# The project runs from /app/.venv, so the system pip is not needed.
RUN python3 -m pip uninstall -y pip

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY scripts ./scripts
COPY LICENSE README.md ./

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["sleep", "infinity"]
