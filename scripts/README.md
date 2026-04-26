# SCAR Training and Evaluation Scripts

The four scripts in this folder train and evaluate SCAR. Each runs on Modal Labs (NVIDIA H100) and is idempotent — checkpoints are written to a shared Modal volume so reruns can resume.

| Script | Stage | Inputs | Outputs |
|---|---|---|---|
| `step5_sae.py` | SAE pretraining | `scar-corpus` | Frozen JumpReLU SAE (Layer 19, 16,384 features) |
| `step6_retrieval.py` | Retrieval fine-tuning | SAE + `scar-pairs` | Backbone LoRA + SAE-LoRA + IDF weights |
| `step8_evmbench.py` | OOD evaluation | Trained retriever | EVMBench P@10, Coverage, MRR |
| `step9_evaluation.py` | Main evaluation | Trained retriever | 838-pair + full-corpus metrics |

## Order of Operations

```bash
# 1. SAE pretraining (~6 H100-hours)
modal run scripts/step5_sae.py

# 2. Retrieval fine-tuning (~2 H100-hours for 25 epochs, rank 256)
modal run scripts/step6_retrieval.py

# 3. Evaluation
modal run scripts/step9_evaluation.py     # 838-pair eval + full-corpus retrieval
modal run scripts/step8_evmbench.py       # OOD evaluation on EVMBench
```

## Step Details

### `step5_sae.py` — SAE Pretraining

Trains a JumpReLU SAE on Layer 19 residual stream activations from Qwen2.5-Coder-1.5B over the 231k-contract corpus.

**Key hyperparameters** (defaults match the published model):

| Parameter | Value |
|---|---|
| SAE width | 16,384 features (10.7× expansion) |
| Target layer | 19 (of 28) |
| Sparsity coefficient (λ_s) | 0.1 (final) |
| Pre-bias coefficient (λ_p) | 3e-6 |
| Learning rate | 2e-4 |
| Batch size | 4,096 tokens (packed) |
| Max sequence length | 1,024 |
| Threshold initialization | 0.1 |
| Target L0 | 37 |

The SAE achieves variance explained ≈ 0.97 at convergence with mean L0 ≈ 37 active features per token. Final checkpoint is uploaded to `scar-weights/sae/checkpoint_final.pt`.

### `step6_retrieval.py` — Retrieval Fine-Tuning

Jointly trains backbone LoRA (rank 64 on Q/K/V/O) and SAE-LoRA (rank 256 on the SAE encoder) for sparse retrieval. Loss combines InfoNCE contrastive + margin-MSE distillation + DF-FLOPS sparsity regularization.

**Key hyperparameters**:

| Parameter | Value |
|---|---|
| Backbone LoRA rank | 64 |
| Backbone LoRA targets | Q, K, V, O projections |
| SAE-LoRA rank | 256 |
| Learning rate | 5e-5 (cosine schedule) |
| Batch size | 32 (in-batch negatives) |
| Epochs | 25 (5,900 steps) |
| Temperature τ | 0.1 |
| Distillation λ | 0.5 (no measurable benefit at 25 ep — see paper) |
| Per-token TopK | 64 |
| Query TopK | 100 |
| Document TopK | 400 |
| Max sequence length | 256 tokens |
| DF-FLOPS λ | 1e-6 |
| Attention | Bidirectional |

The `--epochs` flag controls training length. The published checkpoints are at 15 and 25 epochs. Final checkpoints upload to `scar-weights/scar-15ep/` and `scar-weights/scar-25ep/`.

### `step8_evmbench.py` — Out-of-Distribution Evaluation

Evaluates the trained retriever on EVMBench detect-mode tasks (82 high-severity findings across 22 real audit contests). Indexes the full 231k-contract corpus and retrieves over it.

Outputs P@10, Coverage (fraction of findings where any relevant file appears in top-10), and MRR. Hybrid scoring with BM25 (α = 0.5) is computed automatically.

### `step9_evaluation.py` — Main Evaluation

Runs the full evaluation suite:

- **838-pair controlled eval**: Recall@{1,5,10,20}, MRR, nDCG@10 with 95% bootstrap CIs (n=10,000) and paired bootstrap significance tests
- **Full-corpus eval**: 838 queries against 232,107 documents (corpus + injected ground-truth)
- **Baselines**: BM25, Dense (Qwen Layer 19), CodeBERT, E5-base-v2, SCAR (frozen), SPLADE-Qwen
- **Hybrid sweep**: α ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5}

## Modal Setup

All scripts require:
- A Modal Labs account with H100 access (`modal config set-environment` configured)
- A Modal secret named `huggingface-token` with a HuggingFace write token
- The shared volume `scar-sae-training` and `scar-retrieval-training` (created automatically on first run)

See [`README_modal.md`](README_modal.md) for full setup details.

## Compute Budget

| Stage | H100 hours | Approximate cost |
|---|---:|---:|
| SAE pretraining | ~6 | $25 |
| Retrieval fine-tuning (25 ep) | ~2 | $8 |
| Main evaluation | ~3 | $12 |
| EVMBench evaluation | ~2 | $8 |
| Ablations + baselines | ~50 | $200 |
| **Total (paper)** | **~70** | **~$280** |

## Resuming and Debugging

Each script writes intermediate state to its Modal volume. Reruns will resume from the latest checkpoint unless `--restart` is passed. Logs are streamed to stdout and persisted to `runs/<run_name>/training.log`.

For local debugging without Modal, scripts can be invoked with `--local-test` (where supported) but the corpus and full evaluation require GPU memory beyond consumer hardware.
