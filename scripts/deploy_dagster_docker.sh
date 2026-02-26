#!/bin/bash
# Deploy dagster docker image to registry.
# Usage: ./scripts/deploy_dagster_docker.sh

set -e

REPOSITORY=${REPOSITORY:-"stary"}
DOCKER_TAG=${DOCKER_TAG:-"latest"}

PROJECT_IMAGE=$PROJECT_NAME:$DOCKER_TAG
REMOTE_PROJECT_IMAGE=$DOCKER_REGISTRY_URL/$REPOSITORY/$PROJECT_IMAGE
docker tag $PROJECT_IMAGE $REMOTE_PROJECT_IMAGE
docker push $REMOTE_PROJECT_IMAGE

echo "Pushed $REMOTE_PROJECT_IMAGE"
