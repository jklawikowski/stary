# VerifAI/XPU Proof-of-Concept Runbook

This document describes how to run STARY end-to-end against a sample XPU blocker ticket to validate the VerifAI/XPU integration.

## Prerequisites

- STARY installed and configured (see main README)
- `.env` file with:
  - `JIRA_BASE_URL` and `JIRA_TOKEN` set for access to XPU blocker tickets
  - `GITHUB_TOKEN` with access to VerifAI repositories
  - `ALLOWED_REPOS` including the target VerifAI repos
  - `JENKINS_ALLOWED_HOSTS`, `JENKINS_USERNAME`, `JENKINS_PASSWORD` if tickets contain Jenkins URLs
  - Inference backend configured (`COPILOT_GITHUB_TOKEN` or equivalent)

## Selecting a Test Ticket

Choose a representative XPU blocker ticket. Good candidates:

- **KV cache config adjustment** (e.g. BLK-016/AIFQA-402): Fix is reducing `input_len`/`output_len` or increasing TP
- **Missing XPU kernel**: Fix is adding `PYTORCH_ENABLE_XPU_FALLBACK=1`
- **Device type change**: Fix is changing device from `cuda` to `xpu` in JSON config

## Running the Pipeline

```bash
# Single-ticket mode
python -m stary.main https://jira.devtools.intel.com/browse/AIFQA-402
```

## Evaluation Checklist

### TaskReader

- [ ] Correctly identifies the `[VerifAI]` or BLK-series ticket format
- [ ] Extracts the correct target repository URL(s)
- [ ] Identifies XPU error patterns from Jenkins logs (if linked)
- [ ] Produces tasks with correct scope (JSON config edit, env var addition, etc.)

### Planner

- [ ] Successfully clones and explores the target VerifAI/workloads repo
- [ ] Validates task file paths against actual repo structure
- [ ] Produces concrete implementation steps with correct JSON field names
- [ ] Steps reference actual files in the workloads repository

### Implementer

- [ ] Correctly reads existing JSON workload configs
- [ ] Generates valid JSON config changes (proper syntax, field names)
- [ ] Preserves existing config structure while making targeted changes
- [ ] Creates a valid PR with appropriate commit message

## Known Gaps to Watch For

1. **Repository access**: VerifAI repos may be on `intel-innersource` GitHub — ensure token has access
2. **Large Jenkins logs**: XPU CI logs can be very large — the `search_jenkins_log_xpu_errors` tool should be used instead of fetching full logs
3. **Config complexity**: Some workload configs have nested structures — check that the LLM preserves them correctly
4. **Multiple repos**: Some tickets require changes across multiple repos (e.g. workloads + validation framework) — verify the orchestrator handles this

## Capturing Results

Document for each run:

- Input: Ticket ID and URL
- TaskReader output: interpretation + tasks JSON
- Planner output: validated steps
- Implementer output: PR URL or error
- Any prompt-tuning iterations needed

Store results in a `docs/poc_results/` directory for the research deliverable.
