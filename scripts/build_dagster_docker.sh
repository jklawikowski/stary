#!/bin/bash
# Build and deploy dagster docker image.
# Usage: ./scripts/build_dagster_docker.sh

set -e

REPOSITORY=${REPOSITORY:-"stary"}
PROJECT_NAME=${PROJECT_NAME:-"stary"}
DOCKER_TAG=${DOCKER_TAG:-"latest"}

echo "Building dagster webserver image..."
docker build \
    -f dagster/dev/dagster/Dockerfile \
    -t "${PROJECT_NAME}-dagster:${DOCKER_TAG}" \
    .

echo "Building user deployment image..."
docker build \
    -f dagster/dev/user_deployment/Dockerfile \
    -t "${PROJECT_NAME}:${DOCKER_TAG}" \
    .

echo "Done."
