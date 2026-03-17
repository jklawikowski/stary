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

# Needed by k8s
RUN mkdir -p /opt/dagster && chown $USER_UID:$USER_GID /opt/dagster

USER $USER_NAME
WORKDIR /app

ENV PATH="/home/$USER_NAME/.local/bin:$PATH"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ARG PIP_EXTRA_INDEX_URL="https://af01p-igk.devtools.intel.com/artifactory/api/pypi/qaplatform-igk-local/simple"

RUN --mount=type=cache,mode=0777,uid=$USER_UID,gid=$USER_GID,target=/home/$USER_NAME/.cache/pip \
    pip3 install --user --upgrade pip==24.3.1

COPY ./requirements/ /requirements/
RUN --mount=type=cache,uid=$USER_UID,gid=$USER_GID,mode=0777,target=/home/$USER_NAME/.cache/pip \
    pip3 install -r /requirements/dagster.txt

ARG PROJECT_VERSION
ARG PIP_PACKAGE_NAME=stary
ARG NETRC="/home/$USER_NAME/.netrc"
RUN --mount=type=secret,id=netrc,uid=$USER_UID,target=/home/$USER_NAME/.netrc \
    --mount=type=cache,uid=$USER_UID,gid=$USER_GID,mode=0777,target=/home/$USER_NAME/.cache/pip \
    pip3 install --user $PIP_PACKAGE_NAME==$PROJECT_VERSION

COPY ./dagster/logging.json .
EXPOSE 4000

CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4000", "-m", "stary.dagster.definitions"]
