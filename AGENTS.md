# Repository Guidelines

## Project Structure & Module Organization
Core training code lives in `threedgrut/`, grouped by domain (`model/`, `datasets/`, `strategy/`, `utils/`) and orchestrated via `trainer.py`. Real-time ray tracing kernels reside in `threedgrt_tracer/` and `threedgut_tracer/`; they JIT-compile CUDA/Slang sources under `src/` and `include/`. Hydra configurations in `configs/` (notably `apps/*.yaml` entrypoints) stitch together datasets, strategies, and render settings. Supporting utilities such as dataset prep (`preprocess/`), benchmarking scripts (`benchmark/`), and the interactive GUI (`threedgrut_playground/` and `playground.py`) sit alongside assets and vendor code within `thirdparty/`. Keep data under `data/` and outputs in `runs/` to match internal tooling.

## Build, Test, and Development Commands
- `./install_env.sh 3dgrut [WITH_GCC11]`: create the conda env with compatible CUDA toolchains; re-run when updating third-party submodules.
- `conda activate 3dgrut && pip install -e .`: install the Python entrypoints for interactive development.
- `python train.py --config-name apps/nerf_synthetic_3dgrt.yaml path=data/... out_dir=runs experiment_name=my_scene`: launch training with Hydra overrides.
- `python render.py --config-name render/offline.yaml checkpoint=runs/...`: render evaluation sequences from a saved checkpoint.
- `python playground.py --scene-config configs/apps/colmap_3dgut.yaml`: open the GUI playground for rapid inspection.

## Environment & Runtime Assumptions
- Primary runtime is a Miniconda environment named `3dgrut`; all Python deps, CUDA toolchains, and third-party builds are installed there.
- Before running or testing, either `conda activate 3dgrut` or prefix commands with `conda run -n 3dgrut ...` to ensure the correct toolchain is used.
- Local testing, training, and rendering should be executed within this environment to match CI and avoid mismatched system packages.

## Coding Style & Naming Conventions
Use Python 3.11+, four-space indentation, and type hints consistent with existing modules. Prefer descriptive CamelCase for classes (`Trainer3DGRUT`) and snake_case for functions, Hydra keys, and config files (`particle_kernel_max_alpha`). Keep module-level constants uppercase. Follow the existing docstring pattern of concise triple-double-quoted summaries. When touching CUDA or Slang kernels, mirror naming already in `include/3dgrt/kernels` and update accompanying comments sparingly but clearly.

## Testing Guidelines
Before submitting, run `conda run -n 3dgrut python train.py --help` to confirm CLI integrity (matches CI). For functional changes, execute at least one short training job using the closest `configs/apps/<dataset>_3dgrt.yaml` and capture PSNR/SSIM from `runs/<experiment>/metrics.json`. Regenerate renders via `python render.py ... render.save_frames=true` when modifying rendering paths, and attach representative outputs or regression comparisons in your PR.

## Commit & Pull Request Guidelines
Commits follow the repository’s terse, imperative headline pattern (e.g., `Fix dependency conflicts (#127)`), must include `Signed-off-by` trailers (`git commit -s ...`), and should scope changes to a single concern. Reference issues or configs impacted directly in the body. Pull requests stay in draft until ready, include a clear summary, reproduction commands, metric deltas, and before/after imagery when applicable. Flag any GPU, compiler, or dataset prerequisites in the description so CI reviewers can reproduce locally.
