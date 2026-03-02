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

## Configuration

All configuration is done through environment variables.  Copy `.env.example`
to `.env` and fill in values, or export them before running.

| Variable | Required | Description |
|---|---|---|
| `INFERENCE_URL` | Yes | LLM chat-completion endpoint |
| `AGENT1_INFERENCE_URL` | No | Override inference URL for Agent 1 |
| `AGENT2_INFERENCE_URL` | No | Override inference URL for Agent 2 |
| `AGENT3_INFERENCE_URL` | No | Override inference URL for Agent 3 |
| `JIRA_BASE_URL` | Yes | Base URL of your Jira instance |
| `JIRA_TOKEN` | Yes | Bearer token for Jira REST API |
| `GITHUB_TOKEN` | Yes | GitHub PAT for cloning and PR creation |
| `DAGSTER_BASE_URL` | No | Base URL of the Dagster webserver UI (see below) |

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
[~jklawiko] stary:wip
Pipeline has been triggered and is currently in progress.
[View live pipeline status|https://dagster.example.com/runs/abc12345-def6-7890-ghij-klmnopqrstuv]
```

**Example Jira WIP comment (without `DAGSTER_BASE_URL` configured):**

```
[~jklawiko] stary:wip
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

## License

MIT
