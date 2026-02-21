# HPC Usage Notes

This folder contains scheduler-oriented wrappers for running the pipeline on clusters.

## Quick start

1. Update paths in `hpc/pipeline_batch.pbs`:
- `PROJECT_ROOT`
- `REPO_DIR`
- `INPUT_DIR`
- `OUTPUT_DIR`
- `GROUND_TRUTH`
- `PYTHON_ENV_ACTIVATE`

2. Submit:

```bash
qsub hpc/pipeline_batch.pbs
```

## How chunked batch processing works

`python -m raga_pipeline.batch` now supports:
- `--max-files`: process only N pending files this run.
- `--progress-file`: JSON checkpoint path.
- `--exit-99-on-remaining`: returns exit code `99` when files remain, for PBS resubmission loops.

The default progress file is:

`<output>/logs/<input_dir_name>_batch_progress.json`

## run_pipeline.sh on HPC

`run_pipeline.sh` no longer hardcodes `/opt/miniconda3`. Configure it with:
- `RAGA_CONDA_SH` (optional explicit `conda.sh` path)
- `RAGA_CONDA_ENV` (default: `raga`)
- `RAGA_SKIP_ENV_ACTIVATE=1` to skip conda activation
- `RAGA_PYTHON_BIN` (default: `python3`)
