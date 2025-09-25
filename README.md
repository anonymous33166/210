# Think In Graph(TIG) system: 

This folder contains the core implementation of a multi-paradigm reasoning benchmark with experiment orchestration, multi-model backends, and RGWL similarity analysis.

- TIG_reasoning_framework.py: Unified reasoning engine supporting CoT, ToT, GoT, AoT, EGoT. Builds a directed reasoning graph, evaluates result/termination criteria, and can export graph metadata/visuals.
- TIG_experiment_system.py: End-to-end experiment runner. Loads datasets, sweeps methods, aggregates results, saves reports/CSVs, and triggers RGWL analysis.
- TIG_multi_model_backend.py: Pluggable backends for Azure OpenAI, OpenAI, DeepSeek, Qwen, Claude. Includes robust fallback.
- TIG_rgwl_kernel.py: Enhanced Reasoning Graph Weisfeiler-Lehman (RGWL) kernel for graph similarity, validation (WL tests), and visualization.
- TIG_enhanced_system.py: Main entry with multiple modes (test/single/benchmark/rgwl/backend).
- Mathematcal_Logic_Puzzles.py, Coding_Tasks.py, Legal_Cases.py, College_Entrance_Examination_Questions.py: Dataset-specific runners.

## Data (single-link download)

Provide reviewers a single URL to a zip file that contains all required datasets. Replace the placeholder link below with your hosted file (GitHub Release asset, Zenodo DOI link, OSF, or Drive shared link).

- Data link (anonymous): https://anonymous.4open.science/r/211-3011/

Download and extract:

1) Open the anonymous link above in your browser
2) Click Download to get the archive
3) Unzip the archive into `project/data` so the files match the list below

Expected files after extraction (under `project/data`):

```text
project/data/
  Mathematical_Logic_Puzzles.json
  Coding_Tasks.json
  Legal_Cases.json
  College_Entrance_Examination.json
```

Notes:
- The loaders print clear hints if a file is missing and point to this README for the single-link download.
- You may include additional datasets; the system will skip unknown names.

## Installation

- Python: 3.9–3.11 recommended
- Install dependencies from repo root:

```bash
pip install -r project/requirements.txt
```

## Quickstart

- Run basic end-to-end tests (backend + single-method + RGWL demo):

```bash
python -m project.TIG_enhanced_system --mode test
```

- Run comprehensive benchmark (preconfigured methods/datasets):

```bash
python -m project.TIG_enhanced_system --mode benchmark
```

- Run dataset-specific experiments:

```bash
python -m project.Mathematcal_Logic_Puzzles --runs 3 --parallel 2 --dataset 24point

python -m project.Coding_Tasks --runs 3 --parallel 1 --quiet

python -m project.Legal_Cases --runs 3 --parallel 1 --quiet

python -m project.College_Entrance_Examination_Questions --runs 3 --parallel 1 --quiet
```

- Optional flags:
  - `--quiet` or `BENCHMARK_QUIET=1` to reduce logs
  - `--limit N` (math runner) to run first N problems

## Backend configuration and fallback

The system attempts Azure OpenAI first and falls back across providers. Environment variables (if available):

- Azure: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION
- OpenAI: OPENAI_API_KEY
- DeepSeek: DEEPSEEK_API_KEY
- Qwen: QWEN_API_KEY
- Claude: CLAUDE_API_KEY

## RGWL kernel outputs

When RGWL analysis is enabled (default in most flows), the system saves kernel and distance matrices and validation reports under `results/...`, for example:

- `rgwl_kernel_matrix_*.npy`
- `rgwl_distance_matrix_*.npy`
- `rgwl_kernel_wltest_*.md`, `rgwl_distance_check_*.md`

You can also call the demo directly:

```bash
python -m project.TIG_enhanced_system --mode rgwl
```

## Results and reports

Experiment artifacts are written to a timestamped directory under `results/`:

- Per-dataset detailed JSON and summaries (CSV)
- Method/run-wise aggregates and a concise paper-style table CSV
- Optional graph visualizations (if enabled)
- RGWL matrices and validation reports

## Repository structure (key files)

```text
project/
  TIG_reasoning_framework.py
  TIG_experiment_system.py
  TIG_multi_model_backend.py
  TIG_rgwl_kernel.py
  TIG_enhanced_system.py
  Mathematcal_Logic_Puzzles.py
  Coding_Tasks.py
  Legal_Cases.py
  College_Entrance_Examination_Questions.py
  config/
    azure_config.py
    dynamic_config.json
  data/
    (datasets placed here via single-link download)
  results/
  README.md  (this file)
```

