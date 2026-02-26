ARG BASE_IMAGE=ger-is-registry-pre.caas.intel.com/qa-platform/base/ubuntu-intel:22.04@sha256:8353ed6c7075572269986e8c60c9adae6fed80fcb9fc392d03eb534de058e0a2
FROM $BASE_IMAGE

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Corporate proxy
ENV HTTP_PROXY=http://proxy-dmz.intel.com:912/
ENV http_proxy=http://proxy-dmz.intel.com:912/
ENV HTTPS_PROXY=http://proxy-dmz.intel.com:912/
ENV https_proxy=http://proxy-dmz.intel.com:912/
ENV NO_PROXY=0.0.0.0,127.0.0.1,localhost,intel.com,habana-labs.com
ENV no_proxy=0.0.0.0,127.0.0.1,localhost,intel.com,habana-labs.com

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
# Needed for Dagster compute logs and instance config
RUN mkdir -p /opt/dagster/dagster_home && chown -R $USER_UID:$USER_GID /opt/dagster

USER $USER_NAME
WORKDIR /app

ENV PATH="/home/$USER_NAME/.local/bin:$PATH"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip3 install --user --upgrade pip setuptools wheel

# Install base requirements (includes dagster-postgres for instance storage)
COPY --chown=$USER_UID:$USER_GID requirements/ /requirements/
RUN pip3 install --user -r /requirements/dagster.txt

# Install stary package (editable for dev)
COPY --chown=$USER_UID:$USER_GID . /app
RUN pip3 install --user -e /app

EXPOSE 4000

CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4000", "-m", "stary.dagster.definitions"]

HEALTHCHECK CMD ["dagster", "api", "grpc-health-check", "-h", "0.0.0.0", "-p", "4000"]
