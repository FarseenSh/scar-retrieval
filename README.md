# SCAR: Sparse Code Audit Retriever

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/paper-OpenReview-b31b1b.svg)](https://openreview.net/forum?id=moD8Hxq9hN)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/Farseen0/scar-weights)
[![Datasets](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-yellow)](https://huggingface.co/Farseen0)

> The first sparse latent retriever for smart contract security auditing — built on **SAE-LoRA**, a parameter-efficient adaptation of frozen Sparse Autoencoder features that turns reconstruction-oriented latents into retrieval-discriminative ones.

---

## What is SCAR

SCAR retrieves vulnerable Solidity code given a natural-language audit finding (e.g. *"HIGH severity: price oracle manipulation via flash loan"*). It operates in **SAE latent space** (16,384 features) rather than vocabulary space (152k logits) or dense embedding space, producing inverted-index-compatible sparse vectors that are interpretable, compact, and fast.

The technical contribution is **SAE-LoRA**: a low-rank adaptation of the frozen SAE encoder weights that adapts which features fire during retrieval without modifying the SAE's learned feature dictionary. With only 4.6M additional parameters (~0.3% of the backbone), SAE-LoRA improves standalone R@10 from 0.026 (frozen SAE) to **0.977** on controlled evaluation — a 37.6× improvement — and maintains R@10=0.901 when retrieving against the full 232k-document corpus, while BM25 collapses to 0.308.

## Headline Results

| Metric | BM25 | SPLADE-Qwen | **SCAR-25ep** | **SCAR-15ep** |
|---|---:|---:|---:|---:|
| R@10 (838-pair eval) | 0.689 | 0.963 | **0.977** | 0.971 |
| R@10 (full 232k corpus) | 0.308 | 0.838 | **0.901** | 0.868 |
| MRR (full corpus) | 0.282 | 0.716 | **0.803** | 0.771 |
| nDCG@10 (full corpus) | 0.288 | 0.743 | **0.825** | 0.792 |
| EVMBench coverage (OOD) | 0.720 | — | 0.683 | **0.732** |

All gains over BM25 statistically significant at *p* < 0.0001 (paired bootstrap, n = 10,000).

## Quick Start

```python
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Pull weights once (cached on subsequent calls)
local_dir = snapshot_download("Farseen0/scar-weights")
variant = "scar-25ep"   # or "scar-15ep" for better OOD coverage

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
backbone = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B", torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(backbone, f"{local_dir}/{variant}/lora_adapter")

ckpt = torch.load(f"{local_dir}/{variant}/checkpoint_final.pt", map_location="cpu")
sae_lora_state = ckpt["sae_lora_state"]
idf_weights    = ckpt["idf_weights"]
config         = ckpt["config"]
```

A weight-loading and inspection demo is in [`examples/inspect_weights.py`](examples/inspect_weights.py). For the full encoder used to produce the paper numbers — including bidirectional attention, activation normalization, and the sparse-vector pipeline — see [`scripts/step9_evaluation.py`](scripts/step9_evaluation.py).

## Architecture

```
Input text
    │
    ▼
Qwen2.5-Coder-1.5B + LoRA (rank 64 on Q/K/V/O)
    │
    ▼ Layer 19 residual stream (1536-dim, bidirectional)
    │
JumpReLU SAE encoder (W_e + A·B  ←  SAE-LoRA, rank 256)
    │
    ▼ 16,384 latent features
    │
Per-token TopK (k=64)  →  Sum-pool  →  log1p saturation
    │
    ▼
IDF weighting  →  Document TopK (q=100, d=400)  →  L2 norm
    │
    ▼
Sparse retrieval vector  (~115 active dims, inverted-index compatible)
```

| Component | Spec |
|---|---|
| Backbone | Qwen2.5-Coder-1.5B (28 layers, hidden 1536) |
| SAE | JumpReLU, 16,384 features (10.7× expansion), Layer 19 |
| Backbone LoRA | rank 64 on Q/K/V/O — 17.4M params |
| SAE-LoRA | rank 256 on encoder W_e — 4.6M params |
| Pooling | Sum-pool + log1p saturation |
| Sparsity | Per-token TopK=64; doc TopK=400; query TopK=100 |
| Total trainable | ~22M (1.5% of backbone) |

## Repository Structure

```
scar-retrieval/
├── data/                       # Dataset construction (Modal Labs)
│   ├── pipeline.py             #   Original 9-source pipeline (corpus + pairs + eval)
│   ├── data_pipeline_v2.py     #   FORGE-Curated + FORGE-Artifacts extraction
│   ├── scar_data_pipeline.py   #   Eval expansion + SCAR dataset push
│   └── scar_solodit_api.py     #   Solodit API ingestion
├── scripts/                    # Training and evaluation (Modal Labs)
│   ├── step5_sae.py            #   SAE pretraining (JumpReLU, Layer 19)
│   ├── step6_retrieval.py      #   Retrieval fine-tuning (SAE-LoRA + backbone LoRA)
│   ├── step8_evmbench.py       #   EVMBench OOD evaluation
│   └── step9_evaluation.py     #   Main evaluation (838-pair + full-corpus)
├── utils/                      # Shared utilities
│   ├── hf_utils.py
│   ├── pair_builder.py
│   └── solidity_utils.py
├── examples/
│   └── inspect_weights.py      # Load + inspect weights from HuggingFace
├── requirements.txt
├── LICENSE                     # Apache 2.0
└── CITATION.cff
```

## Reproducing the Paper

The full training pipeline runs on Modal Labs (NVIDIA H100). Total compute: ~70 H100-hours.

```bash
# 1. Build datasets (one-time, ~12 hours total)
modal run data/pipeline.py             # 9-source corpus + base pairs
modal run data/data_pipeline_v2.py     # FORGE extension
modal run data/scar_solodit_api.py     # Solodit API ingestion + final scar-pairs
modal run data/scar_data_pipeline.py   # Eval expansion (scar-eval, 838 pairs)

# 2. SAE pretraining (~6 hours)
modal run scripts/step5_sae.py

# 3. Retrieval fine-tuning (~2 hours for 25 epochs)
modal run scripts/step6_retrieval.py

# 4. Evaluation
modal run scripts/step9_evaluation.py     # 838-pair + full-corpus
modal run scripts/step8_evmbench.py       # EVMBench OOD
```

Each `data/` and `scripts/` subdirectory has its own README with detailed CLI flags and the order of operations. See:
- [`data/README.md`](data/README.md)
- [`scripts/README.md`](scripts/README.md)
- [`scripts/README_modal.md`](scripts/README_modal.md) — Modal Labs setup notes

## Datasets and Weights

All public on HuggingFace:

| Resource | HuggingFace |
|---|---|
| Model weights (sae + scar-25ep + scar-15ep) | [`Farseen0/scar-weights`](https://huggingface.co/Farseen0/scar-weights) |
| Retrieval corpus (231,269 contracts) | [`Farseen0/scar-corpus`](https://huggingface.co/datasets/Farseen0/scar-corpus) |
| Training pairs (7,552 curated) | [`Farseen0/scar-pairs`](https://huggingface.co/datasets/Farseen0/scar-pairs) |
| Evaluation set (838 held-out) | [`Farseen0/scar-eval`](https://huggingface.co/datasets/Farseen0/scar-eval) |

## Limitations

- **Single backbone**: only Qwen2.5-Coder-1.5B is verified; transfer to other code models is untested.
- **OOD generalization**: the 25-epoch model under-covers EVMBench vs BM25 standalone — use the 15-epoch checkpoint or BM25 hybrid for open-domain deployment.
- **Solidity / EVM only**: other smart contract languages (Move, Sway, Vyper, Cairo) are out of distribution.
- **Single-contract granularity**: the indexer treats each contract as one document; cross-contract vulnerabilities may rank below their per-file evidence.

## Roadmap

The current release covers the techniques and results in the EMNLP submission. Next directions, ordered by user-facing impact:

1. **Auditor-tool integrations** — Foundry, Hardhat, and Slither plugins that surface SCAR retrievals inline during real audit workflows.
2. **Cross-language transfer** — extend SAE-LoRA to Move, Sway, Vyper, and Cairo using the same backbone-frozen pipeline; verify whether SAE-LoRA generalizes across smart contract languages.
3. **Live audit-feedback loop** — collect anonymized auditor "was this retrieval relevant?" signals and refine SAE-LoRA without retraining the backbone, turning each audit into training data.
4. **Cross-contract reasoning** — extend the indexer beyond single-file granularity to capture vulnerabilities that span multiple contracts (currently a known limitation, see §6 of the paper).
5. **Larger backbones** — evaluate whether SAE-LoRA's parameter efficiency holds at 7B and 32B parameter scales.

## Citation

```bibtex
@inproceedings{shaikh2026scar,
  title  = {SCAR: Sparse Code Audit Retriever via SAE-LoRA Adaptation},
  author = {Shaikh, Farseen},
  year   = {2026},
  note   = {Under review at EMNLP 2026 (ACL ARR March cycle)},
  url    = {https://openreview.net/forum?id=moD8Hxq9hN}
}
```

## Contact

- **Author**: Farseen Shaikh — `farseenshaikh20@gmail.com`
- **HuggingFace**: [`@Farseen0`](https://huggingface.co/Farseen0)
- **Issues / questions**: please open a GitHub issue on this repository.

## License

[Apache License 2.0](LICENSE) — free for research and commercial use with attribution.

## Acknowledgements

Built on Qwen2.5-Coder by Alibaba and JumpReLU SAEs by Rajamanoharan et al. (2024). Compute provided by Modal Labs. Data sourced from the public audit ecosystem: Solodit, MSC, FORGE-Curated, FORGE-Artifacts, DeFiHackLabs, EVuLLM, SmartBugs-Curated, DeFiVulnLabs, and DISL.
