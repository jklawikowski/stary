#!/bin/bash

set -e

# Setup kubectl
kubectl config set-cluster ger-cluster --server=$K8S_CLUSTER_URL
kubectl config set-credentials sys_qaplatformbot --username=$K8S_USERNAME --password=$K8S_PASSWORD
kubectl config set-context qaplatform --cluster=ger-cluster --user=sys_qaplatformbot --namespace=qa-platform
kubectl config use-context qaplatform

IMAGE_PULL_SECRET="ger-is-registry"

REPOSITORY=$DOCKER_REGISTRY_URL/$REPOSITORY/$PROJECT_NAME
sed -i 's/$DOCKER_TAG/'$DOCKER_TAG'/' ./dagster/prod/user_deployment/values_custom.yaml
sed -i 's/$IMAGE_PULL_SECRET/'$IMAGE_PULL_SECRET'/' ./dagster/prod/user_deployment/values_custom.yaml
sed -i "s#\$REPOSITORY#$REPOSITORY#g" ./dagster/prod/user_deployment/values_custom.yaml
helm upgrade --install user-code dagster/dagster-user-deployments -f ./dagster/prod/user_deployment/values_base.yaml -f ./dagster/prod/user_deployment/values_custom.yaml --version "1.3.14" --skip-schema-validation 
sed -i 's/'$DOCKER_TAG'/$DOCKER_TAG/' ./dagster/prod/user_deployment/values_custom.yaml
sed -i 's/'$IMAGE_PULL_SECRET'/$IMAGE_PULL_SECRET/' ./dagster/prod/user_deployment/values_custom.yaml
sed -i "s#$REPOSITORY#\$REPOSITORY#g" ./dagster/prod/user_deployment/values_custom.yaml