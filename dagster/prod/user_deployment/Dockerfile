ARG BASE_IMAGE=ger-is-registry-pre.caas.intel.com/qa-platform/base/ubuntu-intel:22.04@sha256:8353ed6c7075572269986e8c60c9adae6fed80fcb9fc392d03eb534de058e0a2
FROM $BASE_IMAGE

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    python3 \
    python3-pip \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ARG USER_NAME=docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USER_NAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash $USER_NAME
RUN mkdir /app && chown $USER_UID:$USER_GID /app

USER $USER_NAME
WORKDIR /app

ENV PATH="/home/$USER_NAME/.local/bin:$PATH"

# Install stary package from built wheel
ARG PROJECT_VERSION
ARG PIP_PACKAGE_NAME=stary
RUN pip3 install --user $PIP_PACKAGE_NAME==$PROJECT_VERSION

ARG GIT_USER_NAME="Stary Automation"
ARG GIT_USER_EMAIL="stary-bot@noreply.github.com"
RUN git config --global user.name "$GIT_USER_NAME" && \
    git config --global user.email "$GIT_USER_EMAIL"

EXPOSE 4000

CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4000", "-m", "stary.dagster.definitions"]

HEALTHCHECK --interval=10s --timeout=10s --retries=5 --start-period=30s \
    CMD ["dagster", "api", "grpc-health-check", "-p", "4000"]
