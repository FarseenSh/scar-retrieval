# SCAR Data Pipelines

The four scripts in this folder build the SCAR datasets from public sources. Each script runs on Modal Labs and pushes its output to HuggingFace.

> **Known issue (September 2026).** The evaluation set produced by these pipelines is not disjoint from the training pairs: all 838 `scar-eval` positives appear as training positives in `scar-pairs`, and 800 of the 838 (query, positive) pairs appear verbatim. The report-level holdout and the SHA-256 check described below operated on the wrong unit (they compared normalised source across a split that shared the underlying findings). Numbers measured on `scar-eval` by models trained on `scar-pairs` measure memorisation. A rebuilt training pool with document-level removal of every evaluation target and hard negative (`scar-pairs-clean`, 6,202 pairs) and a held-out, time-split test set are released with the corrected preprint in October 2026.

| Script | Purpose | Output (HuggingFace) |
|---|---|---|
| `pipeline.py` | Original 9-source pipeline | `scar-corpus`, base pairs, base eval |
| `data_pipeline_v2.py` | FORGE-Curated + FORGE-Artifacts extension | Adds ~700 high-quality pairs |
| `scar_solodit_api.py` | Solodit API ingestion + final pair merge | Final `scar-pairs` (7,552 curated) |
| `scar_data_pipeline.py` | Evaluation expansion (91 → 838 pairs) | `scar-eval`, `scar-pairs-extended` |

## Order of Operations

The pipelines should run in roughly this sequence to reproduce the paper datasets:

```bash
# 1. Build the corpus (231k contracts) and base contrastive pairs
modal run data/pipeline.py
modal run data/pipeline.py --wave 1   # Stream 1 (corpus)
modal run data/pipeline.py --wave 2   # Stream 2 (pairs)
modal run data/pipeline.py --wave 3   # Stream 3 (initial eval)

# 2. Extend with FORGE-Curated and FORGE-Artifacts
modal run data/data_pipeline_v2.py --mode curated      # FORGE-Curated track
modal run data/data_pipeline_v2.py --mode artifacts    # FORGE-Artifacts track
modal run data/data_pipeline_v2.py --mode merge        # merge with Stream 2

# 3. Ingest Solodit API findings and build final scar-pairs (7,552 curated)
modal run data/scar_solodit_api.py

# 4. Build the evaluation set (838 pairs from 10 sources, stratified holdout)
modal run data/scar_data_pipeline.py --mode eval_expand
modal run data/scar_data_pipeline.py --mode merge_push
```

## Data Sources

The corpus aggregates Solidity contracts from:

- **DISL** — decompiled smart contracts (sampled 120k after dedup)
- **slither-audited** — `big-multilabel` split, 47k contracts with vulnerability labels
- **SmartBugs Wild** — 47k real-world deployed contracts
- **FORGE-Curated** — audit report code, oldest 60% by report date
- **DeFiVulnLabs** — 48 vulnerability type demonstrations

Quality filters: drop contracts < 50 lines, drop import-only files (>80% imports), hash-dedup on normalized source (strips comments, whitespace, SPDX, pragma).

The contrastive pairs are drawn from:

- **Solodit** — 11+ audit firms (Trail of Bits, OpenZeppelin, Sherlock, Code4rena)
- **MSC** — audit findings with severity classification (HuggingFace)
- **DeFiHackLabs** — exploit reproduction contracts
- **FORGE-Artifacts** — 27k findings + 81k .sol files (joined to extract VFPs)
- **FORGE-Curated** — middle temporal split (after Stream 1 takes oldest 60%)
- **EVuLLM, SmartBugs-Curated, GitmateAI, msc-audits-with-reasons** — supplementary sources

Each pair: `(query, positive, hard_negative, source, severity, vuln_type)` where `query` is a severity-prefixed finding description, `positive` is the vulnerable code snippet, and `hard_negative` is a different vulnerability from the same protocol.

## Output Datasets

| Dataset | Size | HuggingFace |
|---|---:|---|
| Corpus | 231,269 contracts | [`Farseen0/scar-corpus`](https://huggingface.co/datasets/Farseen0/scar-corpus) |
| Training pairs | 7,552 curated | [`Farseen0/scar-pairs`](https://huggingface.co/datasets/Farseen0/scar-pairs) |
| Extended pairs | 11,961 | [`Farseen0/scar-pairs-extended`](https://huggingface.co/datasets/Farseen0/scar-pairs-extended) |
| Evaluation set | 838 pairs | [`Farseen0/scar-eval`](https://huggingface.co/datasets/Farseen0/scar-eval) |

## Why two pair datasets?

`scar-pairs` (7,552 curated) is what all production checkpoints train on. `scar-pairs-extended` (11,961) includes Solodit API expansions and lower-quality sources that were ablated during development. The paper shows that extending to 17,578 pairs *decreased* R@10 from 0.876 to 0.798 — a clear case of data quality > quantity. The extended set is published for transparency, not as the recommended training data. (That comparison was made on the leaked split and is being re-evaluated; see the Known issue above.)

## Modal Setup

All pipelines depend on:
- A Modal Labs account with H100 access
- A Modal secret named `huggingface-token` containing a HuggingFace write token
- The shared volume `scar-intermediates` (created automatically on first run)

See `../scripts/README_modal.md` for complete Modal setup.

## Reproducibility Notes

- The `pipeline.py` script consumes the `slither-audited` `big-multilabel` config (not the default config); using the default produces a much smaller corpus.
- FORGE-Curated VFP matching uses a regex-based join; the original pipeline yielded only 66 pairs before a regex fix (now 694 pairs from FORGE-Curated alone).
- The 838-pair evaluation set was built with a report-level holdout and a SHA-256 check on normalised source. **That check did not establish disjointness** — see the Known issue at the top. The rebuild (`rebuild_split.py` in the research repository) removes, by document hash, every pair whose positive or hard negative equals any evaluation target, and verifies zero overlap on both axes before publishing.
