#!/bin/bash

set -e

SECRET_ARG=""
if [[ -n "$NETRC" && -f "$NETRC" ]]; then
    SECRET_ARG="--secret id=netrc,src=${NETRC}"
fi

docker build \
    --secret id=netrc,src=${NETRC} \
    --build-arg PROJECT_VERSION \
    --build-arg PIP_PACKAGE_NAME \
    --build-arg http_proxy=$HTTP_PROXY \
    --build-arg https_proxy=$HTTPS_PROXY \
    --build-arg no_proxy=$NO_PROXY \
    --build-arg PIP_EXTRA_INDEX_URL \
    --build-arg BASE_IMAGE \
    -t $DAGSTER_IMAGE_NAME:$DOCKER_TAG \
    -f dagster/prod/user_deployment/Dockerfile \
    .