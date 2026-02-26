# HPC Usage Notes

This folder contains scheduler-oriented wrappers for running the pipeline on clusters.

These scripts are aligned with the older HPC conventions in:

- `capstone misc/scripts/wrap.pbs`
- `capstone misc/scripts/f0shell.pbs`

Specifically: explicit env activation, optional ffmpeg PATH injection, exit-code-driven resubmission, and scheduler-first logging.

## Quick start

1. Update paths in `hpc/pipeline_batch.pbs`:
- `PROJECT_ROOT`
- `REPO_DIR`
- `INPUT_DIR`
- `OUTPUT_DIR` (recommended new folder such as `batch_results_v2`)
- `GROUND_TRUTH`
- `PYTHON_ENV_ACTIVATE`
- `FFMPEG_BIN_DIR` (if ffmpeg is not globally available)
- `MAX_FILES_PER_RUN` (batch chunk size)
- `SELF_PBS_PATH` (absolute path used when resubmitting itself)

2. Confirm GPU resource line in `hpc/pipeline_batch.pbs` matches your need:

- Current default requests one GPU:
	- `#PBS -q gpu`
	- `#PBS -l select=1:ncpus=4:gpus=1:mem=24gb`
	  (Adjust the `gpus` resource name if your scheduler exposes it differently; consult `qstat -Q` or `pbsnodes -a` to list available feature names.)

3. Submit:

```bash
qsub hpc/pipeline_batch.pbs
```

If your cluster account requires a project code (per Ashoka PBS docs), submit with:

```bash
qsub -P <PROJECT_CODE> hpc/pipeline_batch.pbs
```

## Ashoka PBS command cheat-sheet (GPU runs)

- Submit: `qsub -q gpu hpc/pipeline_batch.pbs`
- Status: `qstat`
- Running node details: `qstat -n`
- Full job details: `qstat -f <JOB_ID>`
- Cancel: `qdel <JOB_ID>`
- Queue info: `qstat -Q`

Note: login nodes are for edit/submit/debug only; long runs should be through scheduler jobs.

## How chunked batch processing works

`python -m raga_pipeline.batch` now supports:
- `--max-files`: process only N pending files this run.
- `--progress-file`: JSON checkpoint path.
- `--exit-99-on-remaining`: returns exit code `99` when files remain, for PBS resubmission loops.

The default progress file is:

`<output>/logs/<input_dir_name>_batch_progress.json`

## Current wrapper behavior (`hpc/pipeline_batch.pbs`)

- Always runs from `REPO_DIR` so `python -m raga_pipeline.batch` resolves correctly.
- Uses `python3 -m raga_pipeline.batch` directly.
- Processes at most `MAX_FILES_PER_RUN` files per submission.
- Writes per-file logs under `<output>/logs/`.
- If files remain, batch returns `99`; wrapper auto-resubmits itself.
- Activates your chosen virtual environment before running.
- Optionally prepends `FFMPEG_BIN_DIR` to `PATH`.
- Prints allocated node + optional `nvidia-smi` output for GPU sanity checks.

## Typical HPC dataset commands (manual mode)

Without wrapper:

```bash
python3 -m raga_pipeline.batch /path/to/dataset \
	--ground-truth /path/to/dataset_gt.csv \
	--output /path/to/output \
	--mode auto \
	--max-files 300 \
	--exit-99-on-remaining
```

Detect-only pass:

```bash
python3 -m raga_pipeline.batch /path/to/dataset \
	--output /path/to/output \
	--mode detect \
	--max-files 300 \
	--exit-99-on-remaining
```

## run_pipeline.sh on HPC

`run_pipeline.sh` no longer hardcodes `/opt/miniconda3`. Configure it with:
- `RAGA_CONDA_SH` (optional explicit `conda.sh` path)
- `RAGA_CONDA_ENV` (default: `raga`)
- `RAGA_SKIP_ENV_ACTIVATE=1` to skip conda activation
- `RAGA_PYTHON_BIN` (default: `python3`)
