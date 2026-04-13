# Stary – Software Task Agentic Resolution sYstem

Stary is an AI-powered pipeline that reads Jira tickets, implements features
via LLM-generated code, creates pull requests, and performs automated code
reviews.

## Architecture

```
Sensor (Jira poll) → Agent 1 (Jira Reader) → Agent 2 (Implementer) → Agent 3 (Reviewer)
```

The pipeline can run either:
- **Standalone** via `python -m stary.main` (orchestrator loop)
- **Dagster-managed** via the `stary_pipeline_with_markers` job (sensor-driven)

### Modular Inference

The LLM inference layer is abstracted via the `stary.inference` module. This
allows switching inference backends without modifying agent code.

```
Agents → InferenceClient (interface) → Backend (Copilot SDK, OpenAI, etc.)
```

To switch backends, set `INFERENCE_BACKEND` environment variable. Currently
supported: `copilot` (default). Additional backends can be added by
implementing the `InferenceClient` protocol.

## Trigger Account

The automation is triggered by mentioning the **faceless/service account**
`sys_qaplatformbot` in a Jira comment.  This is a dedicated service account
suitable for production use — personal accounts must not be used as trigger
identities.

## Configuration

All configuration is done through environment variables.  Copy `.env.example`
to `.env` and fill in values, or export them before running.

| Variable | Required | Description |
|---|---|---|
| `INFERENCE_BACKEND` | No | Inference backend to use (default: copilot) |
| `COPILOT_GITHUB_TOKEN` or `GH_TOKEN` | Yes* | GitHub token for Copilot SDK authentication |
| `COPILOT_MODEL` | No | Model to use (default: gpt-4o) |
| `JIRA_BASE_URL` | Yes | Base URL of your Jira instance |
| `JIRA_TOKEN` | Yes | Bearer token for Jira REST API |
| `GITHUB_TOKEN` | Yes | GitHub PAT for cloning and PR creation |
| `DAGSTER_BASE_URL` | No | Base URL of the Dagster webserver UI (see below) |
| `ALLOWED_REPOS` | No | Comma-separated `owner/repo` or `owner/*` patterns for PR targets |
| `JENKINS_ALLOWED_HOSTS` | No | Comma-separated Jenkins hostnames for log analysis |
| `JENKINS_USERNAME` | No | Jenkins username for authenticated access |
| `JENKINS_PASSWORD` | No | Jenkins password/API token |

*Required when using the `copilot` backend.

### `ALLOWED_REPOS`

Controls which repositories STARY is permitted to create pull requests
against. This is a security guardrail — if the variable is empty or
unset, **all repositories are denied** (fail-closed).

Patterns use `owner/repo` exact matches or `owner/*` org-wide wildcards.
Matching is case-insensitive.

**Example for VerifAI/XPU work:**

```bash
ALLOWED_REPOS=intel-innersource/frameworks.ai.verifai.validation,intel-innersource/frameworks.ai.validation.workloads,intel-innersource/frameworks.ai.pytorch.gpu-models
```

### `DAGSTER_BASE_URL`

When `DAGSTER_BASE_URL` is set (e.g. `https://dagster.example.com` or
`http://localhost:3000`), the WIP comment posted on the Jira ticket will
include a direct clickable link to the live Dagster pipeline run.

The URL is constructed as:

```
{DAGSTER_BASE_URL}/runs/{run_id}
```

**Example Jira WIP comment (with Dagster link):**

```
[~sys_qaplatformbot] stary:wip
Pipeline has been triggered and is currently in progress.
[View live pipeline status|https://dagster.example.com/runs/abc12345-def6-7890-ghij-klmnopqrstuv]
```

**Example Jira WIP comment (without `DAGSTER_BASE_URL` configured):**

```
[~sys_qaplatformbot] stary:wip
Pipeline has been triggered and is currently in progress.
```

If `DAGSTER_BASE_URL` is not set the tool continues to work exactly as
before — the WIP comment is posted without a pipeline link.  This ensures
full backward compatibility.

The value must be a valid `http://` or `https://` URL.  Trailing slashes
are stripped automatically.

## Running Locally

### Docker Compose (recommended)

```bash
cp .env.example .env
# Edit .env with your credentials
docker compose build
docker compose up
```

The Dagster UI will be available at <http://localhost:3000>.

### Standalone orchestrator

```bash
pip install -e ".[dev]"
python -m stary.main                         # continuous sensor loop
python -m stary.main --once                   # single poll cycle
python -m stary.main https://jira.example.com/browse/PROJ-123  # single ticket
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Smoke Test (Docker Compose)

A convenience script is provided to verify that all Docker Compose services
start correctly and reach a healthy state:

```bash
./scripts/test-compose.sh
```

This will build and start all services, wait for them to become healthy
(up to 120 seconds), then tear everything down.  It is useful for local
verification and CI pipelines.

## Proxy Configuration for Docker Environments

If you are behind a corporate HTTP proxy (common in enterprise environments),
Docker containers may inherit `HTTP_PROXY` / `HTTPS_PROXY` from the host
environment or from Docker daemon settings (`~/.docker/config.json`).  This
can cause **container healthchecks to fail** because requests to `localhost`
get routed through the proxy instead of connecting directly.

### Symptoms

- `docker compose up` fails with:
  ```
  dependency failed to start: container stary-user-deployment is unhealthy
  ```
- The Dagster gRPC code server inside the container starts fine (logs show
  it listening on port 4000), but the healthcheck never passes.

### Solution

The `docker-compose.yaml` already sets `NO_PROXY` to include `localhost`
and `127.0.0.1` for all services.  The healthcheck uses Dagster's built-in
`dagster api grpc-health-check` command which does not go through HTTP
proxy settings.

If you still experience issues:

1. **Check Docker daemon proxy config:**
   ```bash
   cat ~/.docker/config.json
   # Look for "proxies" section
   ```
   Ensure `noProxy` includes `localhost,127.0.0.1`.

2. **Override proxy variables in `.env`:**
   ```bash
   NO_PROXY=0.0.0.0,127.0.0.1,localhost,intel.com,habana-labs.com,dagster-db,dagster,user-deployment,host.docker.internal
   ```

3. **Unset proxy variables** if you don't need them:
   ```bash
   unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
   docker compose up
   ```
