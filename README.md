# CUDA-arch-optimized conda recipes (v1 / rattler-build)

v1 (`recipe.yaml`) recipes for three Dao-AILab / FLA projects, reworked to ship
**per-GPU-generation** and **per-CUDA-version** binary variants instead of one fat
binary, using the new [`__cuda_arch`](https://github.com/conda/ceps/blob/main/cep-0046.md)
virtual package (CEP-46).

```
causal-conv1d/            recipe.yaml + variants.yaml        (CUDA, arch matrix)
flash-attn/               recipe.yaml + variants.yaml + ...  (CUDA, arch matrix, 3 outputs)
fla-core/                 recipe.yaml                        (noarch: python; the `fla` core)
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
# single build. CONDA_OVERRIDE_CUDA injects the __cuda virtual package so
# `pytorch *cuda*` resolves on a host without an NVIDIA driver:
CONDA_OVERRIDE_CUDA=12.9 rattler-build build --recipe causal-conv1d \
  -m pinning/conda_build_config.yaml -m causal-conv1d/variants.yaml \
  --target-platform linux-64 \
  --variant cuda_compiler_version=12.9 \
  --variant cuda_arch=sm90 \
  --variant torch_cuda_arch_list=9.0+PTX \
  --variant min_cuda_arch=9.0 \
  --variant arch_priority=1 \
  --variant python=3.12 \
  --test skip

# flash-linear-attention is noarch — no pinning/variants needed:
rattler-build build --recipe flash-linear-attention
```

> **Building without a GPU** (CI runners, laptops): the build host usually has no
> NVIDIA driver, so the `__cuda` virtual package is absent and `pytorch *cuda*` fails
> to solve. Export `CONDA_OVERRIDE_CUDA=<cuda version>` (matching `cuda_compiler_version`)
> to inject it. If you also run the tests, export `CONDA_OVERRIDE_CUDA_ARCH=<min cc>`
> (e.g. `9.0` for `sm90`) so the CEP-46 `__cuda_arch` run-gate is satisfied — though the
> compiled kernels still can't actually execute without a real GPU. The Action sets both
> for you.

## Build everything and publish to prefix.dev

`.github/workflows/build-and-upload-all.yml` (manual `workflow_dispatch`) fans the full
matrix out into one job per `(package × cuda × arch × python)` cell — 66 jobs total
(32 causal-conv1d + 32 flash-attn + noarch fla-core + noarch flash-linear-attention, with
`sm100×12.6` excluded) — builds each (`--test skip`, CPU runner) and uploads its `.conda` to the
**`cuda-optimized`** channel on prefix.dev with `--skip-existing`. Inputs let you limit to
one package or force-overwrite.

One-time setup before the first run:

```bash
# 1. create the channel at https://prefix.dev/channels  (name: cuda-optimized)
# 2. store an API key with write access as a repo secret:
gh secret set PREFIX_API_KEY --repo wolfv/cuda-packages   # paste the token when prompted
```

Then trigger it:

```bash
gh workflow run build-and-upload-all.yml --repo wolfv/cuda-packages           # all packages
gh workflow run build-and-upload-all.yml --repo wolfv/cuda-packages -f only_package=causal-conv1d
```

> The flash-attn cells are a heavy CUDA compile on CPU-only GitHub-hosted runners and may be
> slow or hit the 6h job limit — consider larger/self-hosted runners for the full matrix.

With `fla-core` now in the channel, all four packages resolve from `cuda-optimized` layered
on `conda-forge` (every other dependency — pytorch, cuda-\*, transformers, einops, triton —
already lives on conda-forge).

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
  Its upstream `setup.py` hard-codes `-gencode` flags for every sm, which makes torch
  ignore `TORCH_CUDA_ARCH_LIST` and build a fat all-arch binary; `patches/0002-…` removes
  them so each variant compiles only its target arch. (flash-attn needs no such patch — it
  already defers arch selection to torch.)
- **flash-attn** — translated from the v0 `meta.yaml`. Builds once via a `staging` output
  and splits into `flash-attn`, `flash-attn-fused-dense`, `flash-attn-layer-norm`; the two
  extension packages pin `flash-attn` exactly, so they inherit its `__cuda_arch` gate.
  `pyproject.toml` / `setup.py` / `LICENSE_CUTLASS.txt` are vendored from the feedstock.
- **flash-linear-attention** — `noarch: python`. Triton kernels JIT at runtime, so there
  is nothing to compile per arch/CUDA version. As of 0.5.0 it ships the high-level
  `fla.layers` / `fla.models` while the core ops live in **`fla-core`** (same `fla`
  namespace, no file overlap — verified against both wheels).
- **fla-core** — `noarch: python`; provides the `fla` namespace (`fla/__init__.py` + the
  Triton `fla.ops`). Deps are just `pytorch >=2.7.0` + `einops` (both on conda-forge).
  Packaged here because it's not on conda-forge; publishing it to `cuda-optimized` is what
  makes flash-linear-attention installable.
