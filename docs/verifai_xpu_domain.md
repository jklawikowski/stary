# VerifAI / XPU Domain Knowledge Reference

This document captures domain-specific knowledge for the VerifAI validation
framework and Intel XPU test infrastructure. It serves as a reference for
understanding test structure, ticket conventions, common error patterns, and
established fix strategies.

---

## 1. VerifAI Test Framework Structure

### JSON Workload Config Files

Each test workload is defined by a JSON configuration file containing the
following key fields:

| Field | Description |
|---|---|
| `exec_bin` | Path to the Python executable or wrapper script used to run the workload |
| `workload_base_cmd` | Base command template for launching the workload |
| `workload_params` | Dictionary of parameter overrides (e.g., model path, batch size, precision) |
| `pre_actions` | List of shell commands executed before the workload (env setup, conda activation) |
| `output_dir` | Directory where test artifacts and logs are written |
| `extra_tests_path` | Optional path to additional test definitions outside the main suite |

### `run.sh` Scripts

Each test suite includes a `run.sh` entry-point script that orchestrates
test execution — parsing the JSON config, setting up the environment via
`pre_actions`, and invoking the workload command.

### Test Naming Convention

Tests follow a structured naming pattern:

```
test__inference__<model>__<config>__<precision>__<mode>__<device>__<backend>
```

For example:

```
test__inference__llama2-7b__default__fp16__eager__xpu__ipex
```

### Repository Locations

| Repository | Purpose |
|---|---|
| `frameworks.ai.verifai.validation` | Core VerifAI validation framework and test harness |
| `frameworks.ai.validation.workloads` | Workload definitions and JSON configs |
| `frameworks.ai.pytorch.gpu-models` | PyTorch model implementations targeting GPU/XPU |

---

## 2. XPU Blocker Ticket Format (BLK-series)

XPU blocker tickets track critical failures that prevent models from running
on Intel XPU hardware. They use the **BLK-NNN** identifier pattern (e.g.,
`BLK-042`).

### Required Sections

1. **Summary** — One-line description of the blocking issue.
2. **Affected Models** — Table listing impacted models and configurations.
3. **Log Evidence** — Relevant log snippets showing the failure.
4. **Resolution** — Description of the fix or workaround applied.

### Affected Models Table Format

| Model | Config | Precision | Error Type | Status |
|---|---|---|---|---|
| llama2-7b | default | fp16 | OOM | Open |
| falcon-40b | default | bf16 | CCL timeout | Resolved |

---

## 3. VerifAI Ticket Conventions

All VerifAI-related tickets use the **`[VerifAI]`** prefix in the summary
line:

```
[VerifAI] Add XPU support for Mamba model inference
```

### Standard Ticket Structure

- **Motivation** — Why this change is needed (e.g., new model enablement,
  failure remediation, performance gap).
- **Definition of Done (DoD)** — Concrete acceptance criteria that must be
  satisfied to close the ticket.
- **Target Repos** — One or more repositories identified in the ticket body
  where changes will land.

---

## 4. Common XPU Error Patterns

The following failure signatures are frequently encountered when running
models on Intel XPU:

### Out of Memory

```
torch.OutOfMemoryError: XPU out of memory
```

GPU memory exceeded during allocation. Fix by reducing batch size or
sequence length (`input_len` / `output_len`), or by increasing tensor
parallelism (TP) to distribute the model across more devices.

### Unified Runtime Resource Exhaustion

```
UR_RESULT_ERROR_OUT_OF_RESOURCES
```

Intel Unified Runtime reports resource exhaustion at the driver level.
Typically caused by excessive concurrent allocations or oversized workloads.

### Missing XPU Backend

```
RuntimeError: PyTorch was compiled without CUDA support
```

The workload is attempting to use CUDA APIs on an XPU-only build. The
`device` field must be changed from `cuda` (or `cpu`) to `xpu`.

### Model Config Incompatibility

```
AttributeError.*rope_parameters
```

The model configuration is missing attributes expected by the runtime
(e.g., RoPE parameter definitions). Requires patching the model config or
updating the model implementation.

### Missing Custom Kernel

```
selective_scan_fwd
```

A custom CUDA kernel (e.g., Mamba's selective scan) has no XPU equivalent.
Requires an XPU-specific kernel implementation or a fallback path.

### Collective Communications Timeout

```
CCL timeout
CCL hang
```

Intel Collective Communications Library (CCL) issues during distributed /
multi-tile runs. Often caused by misconfigured environment variables,
incompatible CCL versions, or network-level problems between ranks.

### Level Zero Driver Error

```
level_zero backend failed
```

Intel Level Zero driver or runtime error. May indicate a driver version
mismatch, firmware issue, or hardware fault.

---

## 5. Common VerifAI Fix Patterns

### Reduce Sequence Lengths for Memory-Constrained XPU

Lower `input_len` and/or `output_len` in the JSON workload config to fit
within available device memory:

```json
{
  "workload_params": {
    "input_len": 128,
    "output_len": 128
  }
}
```

### Increase Tensor Parallelism

Increase the `tensor_parallel` (TP) size to shard the model across more
XPU tiles or devices:

```json
{
  "workload_params": {
    "tensor_parallel": 4
  }
}
```

### Enable XPU Fallback

Add the `PYTORCH_ENABLE_XPU_FALLBACK=1` environment variable so that
unsupported XPU operators fall back to CPU execution:

```json
{
  "pre_actions": [
    "export PYTORCH_ENABLE_XPU_FALLBACK=1"
  ]
}
```

### Change Device to XPU

Update the `device` field from `cuda` or `cpu` to `xpu`:

```json
{
  "workload_params": {
    "device": "xpu"
  }
}
```

### Add XPU-Specific Pre-Actions

Include conda environment activation and XPU-specific environment variable
exports in `pre_actions`:

```json
{
  "pre_actions": [
    "source activate xpu-env",
    "export ZE_AFFINITY_MASK=0",
    "export PYTORCH_ENABLE_XPU_FALLBACK=1"
  ]
}
```

---

## 6. Example JSON Workload Config

A representative workload configuration for running LLaMA 2-7B inference on
XPU:

```json
{
  "exec_bin": "/opt/conda/envs/xpu-env/bin/python",
  "workload_base_cmd": "python run_inference.py",
  "workload_params": {
    "model_name": "llama2-7b",
    "precision": "fp16",
    "device": "xpu",
    "batch_size": 1,
    "input_len": 512,
    "output_len": 128,
    "tensor_parallel": 2
  },
  "pre_actions": [
    "source activate xpu-env",
    "export PYTORCH_ENABLE_XPU_FALLBACK=1",
    "export ZE_AFFINITY_MASK=0,1"
  ],
  "output_dir": "/results/llama2-7b/fp16",
  "extra_tests_path": ""
}
```
