# SCAR — Modal Labs Setup

This document covers the one-time setup required to run any SCAR script (data pipelines, training, evaluation) on Modal Labs.

## Prerequisites

1. **Modal Labs account** with H100 access (academic credits available via [Modal's research program](https://modal.com/research))
2. **HuggingFace account** with a write token
3. Python 3.10+ locally

## One-Time Setup

```bash
# 1. Install the Modal CLI
pip install modal

# 2. Authenticate with Modal (opens a browser)
modal token new

# 3. Create the HuggingFace secret in Modal
#    Get a write token from: https://huggingface.co/settings/tokens
modal secret create huggingface-token HF_TOKEN=hf_your_token_here

# 4. Set your HuggingFace username in the data pipeline
#    Edit data/pipeline.py and change:
#      HF_USERNAME = "YOUR_USERNAME"
#    to your actual HF username
```

## Running Scripts

All scripts are invoked with `modal run` from the repository root:

```bash
# Data pipelines (see data/README.md for full sequence)
modal run data/pipeline.py
modal run data/data_pipeline_v2.py
modal run data/scar_solodit_api.py
modal run data/scar_data_pipeline.py

# Training and evaluation (see scripts/README.md for hyperparameters)
modal run scripts/step5_sae.py
modal run scripts/step6_retrieval.py
modal run scripts/step9_evaluation.py
modal run scripts/step8_evmbench.py
```

## Pipeline Wave Flags (data/pipeline.py)

The original pipeline is structured into three waves:

```bash
modal run data/pipeline.py            # Run all waves end-to-end
modal run data/pipeline.py --wave 1   # Source processing (9 containers in parallel)
modal run data/pipeline.py --wave 2   # Merge, dedup, push to HuggingFace
modal run data/pipeline.py --wave 3   # Validation checks
```

### Wave 1 (~4–5 hours, 9 parallel CPU containers)
Each container processes one data source independently — downloads, parses, deduplicates, saves Parquet to the shared Modal Volume. The bottleneck is typically Solodit (16+ audit-firm markdown formats).

### Wave 2 (~30 minutes, sequential)
Merges intermediates, runs cross-source deduplication, performs leakage checks (SHA-256 on normalized source), and pushes the three datasets to HuggingFace.

### Wave 3 (~5 minutes)
Runs 11 validation checks against the published datasets and prints a dashboard with PASS/FAIL for each.

## Network Disconnect / Resume

Once `modal run` launches the containers, they execute independently in Modal's cloud. If the local machine disconnects:

- **Running containers continue** — they finish their work
- **Results are persisted** to the shared Modal Volume
- **On reconnect**, re-run the same command — skip-if-exists checks detect completed intermediates and resume from the last unfinished step

## Estimated Cost (Data Pipeline)

| Phase | Time | Cost |
|---|---:|---:|
| Wave 1 (9 parallel CPU containers) | ~5 hrs | ~$4–6 |
| Wave 2 (merge + push) | ~30 min | ~$0.50 |
| Wave 3 (validation) | ~5 min | ~$0.05 |
| **Data pipeline total** | **~6 hrs** | **~$5–7** |

For full training + evaluation costs see [`README.md`](README.md).

## Monitoring

- **Modal Dashboard**: <https://modal.com/apps> — running functions, logs, real-time cost
- **Terminal output**: each pipeline prints timestamped progress from every container
- **Volume inspection**: `modal volume ls scar-intermediates`, `modal volume get scar-intermediates <path>`

## Troubleshooting

| Error | Fix |
|---|---|
| `Secret not found: huggingface-token` | `modal secret create huggingface-token HF_TOKEN=hf_xxx` |
| Container timeout | Defaults are generous (4+ hrs); re-run — skip-if-exists resumes |
| HF push fails (`401`) | Verify the token has WRITE permission, confirm `HF_USERNAME` is correct |
| `Volume not found` | Volumes auto-create on first use; re-run the script |

## Output

After successful completion of the data pipeline you will have:

- `YOUR_USERNAME/scar-corpus` on HuggingFace (~231k contracts)
- `YOUR_USERNAME/scar-pairs` on HuggingFace (~7,552 curated pairs)
- `YOUR_USERNAME/scar-eval` on HuggingFace (~838 evaluation pairs)
- `validation_report.json` in the `scar-intermediates` Modal Volume
