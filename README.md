# CUDA-arch-optimized conda recipes (v1 / rattler-build)

v1 (`recipe.yaml`) recipes for three Dao-AILab / FLA projects, reworked to ship
**per-GPU-generation** and **per-CUDA-version** binary variants instead of one fat
binary, using the new [`__cuda_arch`](https://github.com/conda/ceps/blob/main/cep-0046.md)
virtual package (CEP-46).

```
causal-conv1d/            recipe.yaml + variants.yaml        (CUDA, arch matrix)
flash-attn/               recipe.yaml + variants.yaml + ...  (CUDA, arch matrix, 3 outputs)
flash-linear-attention/   recipe.yaml                        (noarch: python, Triton JIT)
pinning/                  conda_build_config.yaml            (vendored conda-forge global pinning)
```

## What `__cuda_arch` gives us (CEP-46)

`__cuda_arch` is a virtual package that reports the **minimum** CUDA compute capability
across all detected GPUs as `{major}.{minor}` (e.g. `9.0`). It is absent when there is no
CUDA device, and is overridable with `CONDA_OVERRIDE_CUDA_ARCH`. This lets a recipe:

1. **gate** a build to hardware that can actually run it (`run: __cuda_arch >=9.0`), and
2. ship **several arch variants** of the same package and let the solver pick the best fit.

> Installing `__cuda_arch`-gated packages requires a client that implements the vpkg —
> the [`conda-incubator/nvidia-virtual-packages`](https://github.com/conda-incubator/nvidia-virtual-packages)
> plugin. Without it, `__cuda_arch` is treated as absent and these variants won't resolve.

## The two variant axes

Both compiled packages multiply two axes (see each `variants.yaml`):

| Axis | Values | Mechanism |
|------|--------|-----------|
| **CUDA version** | 12.6, 12.9, 13.0 | `cuda_compiler_version`; `cuda-version ==X` in host pins the runtime so only the matching build resolves |
| **GPU generation** | sm80 / sm90 / sm100 | `cuda_arch` (+ zipped `torch_cuda_arch_list`, `min_cuda_arch`, `arch_priority`) |

Per generation (the `cuda_arch` label is the group's minimum sm, matching the run-gate):

| `cuda_arch` | Generation | `TORCH_CUDA_ARCH_LIST` | run-gate | priority |
|-------------|------------|------------------------|----------|----------|
| `sm80`  | Ampere (sm_80/86/89)    | `8.0;8.6;8.9+PTX`  | `__cuda_arch >=8.0`  | fallback (penalty 2) |
| `sm90`  | Hopper (sm_90)          | `9.0+PTX`          | `__cuda_arch >=9.0`  | penalty 1 |
| `sm100` | Blackwell (sm_100/120)  | `10.0;12.0+PTX`    | `__cuda_arch >=10.0` | preferred (penalty 0) |

**How selection works.** The `__cuda_arch >= min` run requirement makes a variant
*installable* only on hardware whose minimum device meets that compute capability. When
more than one variant qualifies (e.g. an sm_100 box can run all three via SASS+PTX),
`down_prioritize_variant` (= `arch_priority`) makes the solver prefer the least-penalized
= highest-arch build. So sm_80 → `sm80`, sm_90 → `sm90`, sm_100/120 → `sm100`,
automatically. Each group carries `+PTX` on its top capability for forward compatibility,
per the CEP's usage notes.

`sm100 × CUDA 12.6` is skipped because sm_100/sm_120 require CUDA ≥ 12.8.

## Building

The recipes use conda-forge's global keys (`c_stdlib`, `cuda_compiler`, …), so pass the
vendored pinning **first** and the per-recipe `variants.yaml` **second** (later configs
override earlier ones, so the arch matrix wins):

```bash
# Inspect the full (cuda × arch × python) matrix without building.
# (--target-platform linux-64 because the CUDA packages are linux-only.)
rattler-build build --recipe causal-conv1d \
  -m pinning/conda_build_config.yaml -m causal-conv1d/variants.yaml \
  --target-platform linux-64 --render-only

rattler-build build --recipe flash-attn \
  -m pinning/conda_build_config.yaml -m flash-attn/variants.yaml \
  --target-platform linux-64 --render-only

# Build one specific cell (run on a linux machine with the CUDA stack).
# `cuda_arch` is zipped with three sibling keys, so override the whole tuple
# (overriding cuda_arch alone breaks the equal-length zip). Pin python for a
# single build:
rattler-build build --recipe causal-conv1d \
  -m pinning/conda_build_config.yaml -m causal-conv1d/variants.yaml \
  --target-platform linux-64 \
  --variant cuda_compiler_version=12.9 \
  --variant cuda_arch=sm90 \
  --variant torch_cuda_arch_list=9.0+PTX \
  --variant min_cuda_arch=9.0 \
  --variant arch_priority=1 \
  --variant python=3.12

# flash-linear-attention is noarch — no pinning/variants needed:
rattler-build build --recipe flash-linear-attention
```

## Build a single variant from GitHub Actions

`.github/workflows/build-variant.yml` exposes a manual **`workflow_dispatch`** trigger
("Run workflow" button, or `gh workflow run build-variant.yml`). Pick the `package`,
`cuda_compiler_version`, `cuda_arch` (sm80/sm90/sm100) and `python_version`; it maps the
arch to its zipped tuple, builds that one cell, and uploads the `.conda` as an artifact.
Tests default to off (GitHub-hosted runners have no GPU). Note flash-attn's CUDA compile
is heavy and may strain free runners.

Refresh the vendored pinning any time with:

```bash
curl -L https://raw.githubusercontent.com/conda-forge/conda-forge-pinning-feedstock/main/recipe/conda_build_config.yaml \
  -o pinning/conda_build_config.yaml
```

## Notes per package

- **causal-conv1d** — ported from the existing v1 feedstock recipe; added the arch axis
  (`TORCH_CUDA_ARCH_LIST`, build-string tag, `__cuda_arch` gate, prioritization, skip).
- **flash-attn** — translated from the v0 `meta.yaml`. Builds once via a `staging` output
  and splits into `flash-attn`, `flash-attn-fused-dense`, `flash-attn-layer-norm`; the two
  extension packages pin `flash-attn` exactly, so they inherit its `__cuda_arch` gate.
  `pyproject.toml` / `setup.py` / `LICENSE_CUTLASS.txt` are vendored from the feedstock.
- **flash-linear-attention** — `noarch: python`. Triton kernels JIT at runtime, so there
  is nothing to compile per arch/CUDA version. As of 0.5.0 it is a thin meta package over
  **`fla-core`**, which is **not yet on conda-forge** — that recipe must be added before
  this one can build/test (the `import fla` test needs it).
