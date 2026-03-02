#!/bin/bash
# Smoke test: verify that all Docker Compose services start and become healthy.
#
# Usage:
#   ./scripts/test-compose.sh
#
# The script will:
#   1. Build and start all services in detached mode.
#   2. Wait up to 120 seconds for every service to reach a healthy state.
#   3. Tear down the environment (including volumes).
#   4. Exit 0 on success, non-zero on failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

WAIT_TIMEOUT=${COMPOSE_WAIT_TIMEOUT:-120}

cleanup() {
    echo "Tearing down services..."
    docker compose down -v --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

echo "========================================"
echo " Stary Docker Compose Smoke Test"
echo "========================================"
echo ""
echo "Building images..."
docker compose build

echo ""
echo "Starting services (timeout: ${WAIT_TIMEOUT}s)..."
docker compose up -d --wait --wait-timeout "$WAIT_TIMEOUT"

echo ""
echo "All services healthy!"
echo ""

# Print service status for visibility
docker compose ps

echo ""
echo "========================================"
echo " Smoke test PASSED"
echo "========================================"
