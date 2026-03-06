"""
SCAR Step 8: EVMbench Integration — Modal Labs
=========================================================
Evaluates SCAR's retrieval on EVMbench Detect mode tasks.

For each of the 40 detect-mode audits:
  1. Load the contract source code from the EVMbench repo
  2. Use SCAR to retrieve top-K similar vulnerable contracts from our index
  3. Measure retrieval quality: do retrieved contracts contain vulnerabilities
     similar to the ground truth findings?

This shows downstream utility: better retrieval → more relevant context for
vulnerability detection. We don't run a full LLM agent — we measure whether
our retriever surfaces relevant vulnerability patterns.

Metrics:
  - Retrieval Precision@K: fraction of retrieved docs that match vulnerability type
  - Coverage: fraction of gold findings where at least one relevant doc was retrieved
  - MRR: mean reciprocal rank of first relevant retrieval per finding

Usage:
  modal run scripts/step8_evmbench.py                    # full evaluation
  modal run scripts/step8_evmbench.py --mode setup       # clone repos only
  modal run scripts/step8_evmbench.py --mode retrieve    # retrieval eval only
"""

import modal
import os
import math

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("scar-evmbench")

sae_vol = modal.Volume.from_name("scar-sae-training")
retrieval_vol = modal.Volume.from_name("scar-retrieval-training", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.45.0,<5.0.0",
        "datasets>=2.19.0,<4.0.0",
        "huggingface_hub>=0.23.0",
        "safetensors>=0.4.0",
        "tqdm>=4.66.0",
        "peft>=0.10.0,<1.0.0",
        "rank_bm25>=0.2.2",
        "pyyaml>=6.0",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

SAE_DIR = "/sae_training"
SAVE_DIR = "/retrieval_training"
HF_SECRET = modal.Secret.from_name("huggingface-token")
HF_USERNAME = "Farseen0"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
D_IN = 1536
D_SAE = 16384
LAYER_IDX = 19

EVMBENCH_REPO = "https://github.com/paradigmxyz/evmbench.git"
FRONTIER_EVALS_REPO = "https://github.com/openai/frontier-evals.git"


def log(msg: str):
    import sys
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# SAE Encoder (same as step6/step9)
# ═══════════════════════════════════════════════════════════════════════════

def build_sae_encoder():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class JumpReLUSAEEncoder(nn.Module):
        def __init__(self, d_in=D_IN, d_sae=D_SAE, lora_rank=0):
            super().__init__()
            self.d_in = d_in
            self.d_sae = d_sae
            self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
            self.b_enc = nn.Parameter(torch.zeros(d_sae))
            self.b_dec = nn.Parameter(torch.zeros(d_in))
            self.log_threshold = nn.Parameter(torch.zeros(d_sae))
            self.lora_rank = lora_rank
            if lora_rank > 0:
                self.lora_A = nn.Parameter(torch.empty(d_in, lora_rank))
                self.lora_B = nn.Parameter(torch.zeros(lora_rank, d_sae))
                nn.init.kaiming_uniform_(self.lora_A)

        @property
        def threshold(self):
            return self.log_threshold.exp()

        def encode(self, x):
            x_centered = x - self.b_dec
            if self.lora_rank > 0:
                W_eff = self.W_enc + self.lora_A @ self.lora_B
            else:
                W_eff = self.W_enc
            z_pre = x_centered @ W_eff + self.b_enc
            z = F.relu(z_pre - self.threshold)
            return z, z_pre

    return JumpReLUSAEEncoder


def load_frozen_sae(sae_checkpoint_path, device, sae_lora_rank=0):
    import torch
    SAEEncoder = build_sae_encoder()
    ckpt = torch.load(sae_checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["sae_state_dict"]
    d_in = state["W_enc"].shape[0]
    d_sae = state["W_enc"].shape[1]

    target_norm = math.sqrt(d_in)
    if "normalizer_state_dict" in ckpt:
        norm_state = ckpt["normalizer_state_dict"]
        target_norm = norm_state.get("target_norm", target_norm)

    sae = SAEEncoder(d_in=d_in, d_sae=d_sae, lora_rank=sae_lora_rank).to(device)
    sae.W_enc.data.copy_(state["W_enc"])
    sae.b_enc.data.copy_(state["b_enc"])
    sae.b_dec.data.copy_(state["b_dec"])
    sae.log_threshold.data.copy_(state["log_threshold"])
    sae.eval()
    for p in [sae.W_enc, sae.b_enc, sae.b_dec, sae.log_threshold]:
        p.requires_grad = False
    return sae, target_norm


# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional Attention + Layer Access
# ═══════════════════════════════════════════════════════════════════════════

def enable_bidirectional_attention(model):
    import torch
    import types

    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        try:
            qwen_model = model.base_model.model.model
        except AttributeError:
            qwen_model = model.base_model
    else:
        qwen_model = model.model

    def _bidirectional_update_causal_mask(self, attention_mask, input_tensor,
                                          cache_position, past_key_values,
                                          output_attentions=False):
        batch_size, seq_len = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device
        mask_4d = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=dtype, device=device)
        if attention_mask is not None and attention_mask.dim() == 2:
            padding_mask = attention_mask[:, None, None, :].to(dtype)
            min_val = torch.finfo(dtype).min
            mask_4d = mask_4d.masked_fill(padding_mask == 0, min_val)
        return mask_4d

    qwen_model._update_causal_mask = types.MethodType(
        _bidirectional_update_causal_mask, qwen_model
    )


def _get_target_layer(model, layer_idx):
    try:
        return model.base_model.model.model.layers[layer_idx]
    except (AttributeError, TypeError):
        pass
    try:
        return model.model.layers[layer_idx]
    except (AttributeError, TypeError):
        pass
    raise ValueError(f"Cannot find layer {layer_idx}")


# ═══════════════════════════════════════════════════════════════════════════
# Retriever Class (same as step6/step9)
# ═══════════════════════════════════════════════════════════════════════════

def build_retriever_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ScarRetriever(nn.Module):
        def __init__(self, model, tokenizer, sae, layer_idx=LAYER_IDX,
                     topk_query=100, topk_doc=400, max_seq_len=256,
                     target_norm=None, per_token_k=64):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self.sae = sae
            self.layer_idx = layer_idx
            self.topk_query = topk_query
            self.topk_doc = topk_doc
            self.max_seq_len = max_seq_len
            self.target_norm = target_norm
            self.per_token_k = per_token_k
            self._activations = None
            self.idf_weights = None

        def _encode_texts(self, texts, topk):
            import torch

            device = next(self.model.parameters()).device
            enc = self.tokenizer(
                texts, return_tensors="pt", truncation=True,
                max_length=self.max_seq_len, padding=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask_2d = enc["attention_mask"].to(device)

            target_layer = _get_target_layer(self.model, self.layer_idx)

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                self._activations = hidden.float()

            handle = target_layer.register_forward_hook(hook_fn)
            try:
                self.model(input_ids=input_ids, attention_mask=attention_mask_2d,
                           use_cache=False)
            finally:
                handle.remove()

            acts = self._activations
            batch_size, seq_len, d_in = acts.shape

            if self.target_norm is not None:
                with torch.no_grad():
                    batch_mean_norm = acts.norm(dim=-1).mean().clamp(min=1e-8)
                    norm_scale = self.target_norm / batch_mean_norm
                acts = acts * norm_scale

            flat_acts = acts.reshape(-1, d_in)
            z, z_pre = self.sae.encode(flat_acts)

            z_pre_3d = z_pre.reshape(batch_size, seq_len, -1)
            z_pre_pos = F.relu(z_pre_3d)

            tok_topk_vals, tok_topk_idx = torch.topk(
                z_pre_pos, self.per_token_k, dim=-1,
            )
            z_sparse = torch.zeros_like(z_pre_pos).scatter(
                -1, tok_topk_idx, tok_topk_vals,
            )

            pad_mask = attention_mask_2d.unsqueeze(-1).float()
            z_sparse = z_sparse * pad_mask

            pooled_raw = z_sparse.sum(dim=1)
            pooled = torch.log1p(pooled_raw)

            if self.idf_weights is not None:
                pooled = pooled * self.idf_weights.to(pooled.device)

            topk_vals, topk_idx = torch.topk(pooled, topk, dim=-1)
            sparse = torch.zeros_like(pooled).scatter(-1, topk_idx, topk_vals)
            sparse = F.normalize(sparse, dim=-1)

            self._activations = None
            return sparse, pooled

        def encode_queries(self, texts):
            return self._encode_texts(texts, self.topk_query)

        def encode_documents(self, texts):
            return self._encode_texts(texts, self.topk_doc)

        @torch.no_grad()
        def compute_idf(self, documents, batch_size=16):
            import torch

            old_idf = self.idf_weights
            self.idf_weights = None

            N = len(documents)
            d_sae = self.sae.d_sae
            doc_freq = torch.zeros(d_sae)

            for i in range(0, N, batch_size):
                batch = documents[i:i+batch_size]
                _, pooled = self._encode_texts(batch, d_sae)
                active = (pooled > 0).float().cpu()
                doc_freq += active.sum(dim=0)

            idf = torch.log(torch.tensor(N, dtype=torch.float32) / (1.0 + doc_freq))
            idf = idf.clamp(min=0.0)
            self.idf_weights = idf
            return idf

    return ScarRetriever


# ═══════════════════════════════════════════════════════════════════════════
# EVMBench Detect Mode Evaluation
# ═══════════════════════════════════════════════════════════════════════════

# Vulnerability type keywords for matching retrieved docs to findings
VULN_TYPE_KEYWORDS = {
    "reentrancy": ["reentrancy", "reentrant", "re-entrant", "call.value",
                    "external call before state", "recursive call"],
    "access_control": ["access control", "authorization", "onlyowner",
                       "msg.sender", "privilege", "unauthorized",
                       "anyone can call", "permissionless"],
    "integer_overflow": ["overflow", "underflow", "integer", "arithmetic",
                         "truncation", "safecast", "uint96", "unchecked"],
    "price_manipulation": ["price manipulation", "oracle", "flash loan",
                           "price oracle", "price feed", "spot price",
                           "sandwich"],
    "front_running": ["front-run", "frontrun", "mev", "sandwich",
                      "transaction ordering"],
    "logic_error": ["logic error", "incorrect calculation", "wrong formula",
                    "off-by-one", "rounding"],
    "denial_of_service": ["denial of service", "dos", "gas limit", "block gas",
                          "unbounded loop", "out of gas"],
    "data_validation": ["input validation", "parameter check", "zero address",
                        "slippage", "bounds check"],
}


def classify_vulnerability(text):
    """Classify a vulnerability description into vulnerability types."""
    text_lower = text.lower()
    matched_types = []
    for vtype, keywords in VULN_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                matched_types.append(vtype)
                break
    return matched_types if matched_types else ["other"]


@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={
        SAE_DIR: sae_vol,
        SAVE_DIR: retrieval_vol,
    },
    secrets=[HF_SECRET],
    timeout=14400,  # 4 hours
)
def eval_evmbench(
    run_name: str = "primary",
    sae_run_name: str = "primary_L19_16k",
    topk_query: int = 100,
    topk_doc: int = 400,
    lora_rank: int = 64,
    max_seq_len: int = 256,
    top_k_retrieve: int = 10,
    sae_lora_rank: int = 0,
):
    """Run EVMbench Detect mode retrieval evaluation.

    For each of the 40 detect tasks:
    1. Parse config.yaml to get ground truth vulnerabilities
    2. Parse gold_audit.md to get vulnerability descriptions
    3. Collect all .sol files from the audit as the "target contract"
    4. Use vulnerability descriptions as queries
    5. Retrieve top-K similar contracts from our SAE corpus index
    6. Measure: do retrieved contracts contain similar vulnerability patterns?

    Also evaluates BM25 baseline for comparison.
    """
    import torch
    import json
    import yaml
    import subprocess
    from pathlib import Path
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import re

    sae_vol.reload()
    retrieval_vol.reload()

    device = torch.device("cuda")
    hf_token = os.environ.get("HF_TOKEN")

    # --- Clone EVMbench repos ---
    evmbench_dir = "/tmp/evmbench"
    frontier_dir = "/tmp/frontier-evals"

    if not os.path.exists(f"{frontier_dir}/project/evmbench/audits"):
        log("Cloning frontier-evals (contains EVMbench audits)...")
        subprocess.run(
            ["git", "clone", "--depth=1", FRONTIER_EVALS_REPO, frontier_dir],
            check=True, capture_output=True,
        )
        log("frontier-evals cloned")

    audits_dir = Path(frontier_dir) / "project" / "evmbench" / "audits"
    splits_dir = Path(frontier_dir) / "project" / "evmbench" / "splits"

    # --- Parse detect tasks ---
    detect_tasks_file = splits_dir / "detect-tasks.txt"
    if not detect_tasks_file.exists():
        log(f"ERROR: {detect_tasks_file} not found")
        return {}

    detect_task_ids = [
        line.strip() for line in detect_tasks_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    log(f"EVMbench Detect mode: {len(detect_task_ids)} tasks")

    # --- Parse all audit configs and findings ---
    all_tasks = []
    for audit_folder in sorted(audits_dir.iterdir()):
        if audit_folder.name == "template" or not audit_folder.is_dir():
            continue

        config_path = audit_folder / "config.yaml"
        if not config_path.exists():
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)

        audit_id = config.get("id", audit_folder.name)

        # Check if this audit is in detect tasks
        if audit_id not in detect_task_ids:
            continue

        # Parse gold audit (ground truth)
        gold_path = audit_folder / "findings" / "gold_audit.md"
        if not gold_path.exists():
            log(f"  {audit_id}: no gold_audit.md, skipping")
            continue

        gold_text = gold_path.read_text()

        # Parse individual findings
        findings = config.get("vulnerabilities", [])
        finding_texts = []
        for vuln in findings:
            vuln_id = vuln.get("id", "")
            vuln_title = vuln.get("title", "")
            # Try to load individual finding markdown
            finding_path = audit_folder / "findings" / f"{vuln_id}.md"
            if finding_path.exists():
                finding_text = finding_path.read_text()
            else:
                finding_text = f"{vuln_id}: {vuln_title}"
            finding_texts.append({
                "id": vuln_id,
                "title": vuln_title,
                "text": finding_text,
                "vuln_types": classify_vulnerability(f"{vuln_title} {finding_text}"),
            })

        # Collect Solidity source files
        sol_files = list(audit_folder.rglob("*.sol"))
        # Exclude test files and findings
        sol_files = [
            f for f in sol_files
            if "test" not in str(f).lower()
            and "findings" not in str(f).lower()
            and "exploit" not in str(f).lower()
        ]

        contract_code = ""
        for sol_file in sol_files[:20]:  # cap at 20 files per audit
            try:
                code = sol_file.read_text()
                if len(code) > 100:
                    contract_code += f"\n// === {sol_file.name} ===\n{code}\n"
            except UnicodeDecodeError:
                continue

        if not contract_code:
            log(f"  {audit_id}: no Solidity source files found, skipping")
            continue

        all_tasks.append({
            "audit_id": audit_id,
            "contract_code": contract_code[:50000],  # cap at 50k chars
            "findings": finding_texts,
            "gold_text": gold_text,
            "n_sol_files": len(sol_files),
        })

    log(f"Parsed {len(all_tasks)} detect tasks with {sum(len(t['findings']) for t in all_tasks)} total findings")

    if not all_tasks:
        log("ERROR: No valid detect tasks found")
        return {}

    # --- Build corpus index from SAE corpus ---
    log("Loading SAE corpus for retrieval index...")
    corpus_ds = load_dataset(
        f"{HF_USERNAME}/scar-corpus", split="train", token=hf_token
    )
    # Use full corpus for indexing
    corpus_docs = [row["contract_code"] for row in corpus_ds]
    corpus_labels = [row.get("vuln_labels", "") for row in corpus_ds]
    log(f"Corpus: {len(corpus_docs)} documents")

    # --- Load retriever (SCAR) ---
    log("Loading backbone model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Load SAE
    sae_ckpt_path = f"{SAE_DIR}/{sae_run_name}/checkpoint_final.pt"
    if not os.path.exists(sae_ckpt_path):
        log(f"ERROR: SAE checkpoint not found at {sae_ckpt_path}")
        return {}

    sae, target_norm = load_frozen_sae(sae_ckpt_path, device, sae_lora_rank=sae_lora_rank)

    # Load LoRA
    from peft import PeftModel, LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType

    lora_dir = f"{SAVE_DIR}/{run_name}/lora_adapter"
    retrieval_ckpt = f"{SAVE_DIR}/{run_name}/checkpoint_final.pt"

    if os.path.exists(lora_dir):
        log(f"Loading LoRA from {lora_dir}")
        model_with_lora = PeftModel.from_pretrained(backbone, lora_dir)
        model_with_lora.eval()
    elif os.path.exists(retrieval_ckpt):
        log(f"Loading LoRA from checkpoint {retrieval_ckpt}")
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model_with_lora = get_peft_model(backbone, lora_config)
        ckpt = torch.load(retrieval_ckpt, map_location=device, weights_only=False)
        set_peft_model_state_dict(model_with_lora, ckpt["model_state_dict"])
        model_with_lora.eval()
    else:
        log(f"WARNING: No LoRA checkpoint found, using bare backbone")
        model_with_lora = backbone

    enable_bidirectional_attention(model_with_lora)

    # Load SAE LoRA weights from retrieval checkpoint
    if sae_lora_rank > 0 and os.path.exists(retrieval_ckpt):
        ret_ckpt = torch.load(retrieval_ckpt, map_location=device, weights_only=False)
        if "sae_lora_state" in ret_ckpt:
            sae.load_state_dict(ret_ckpt["sae_lora_state"], strict=False)
            log(f"SAE LoRA weights loaded (rank={sae_lora_rank})")
        else:
            log(f"WARNING: no sae_lora_state in checkpoint for SAE LoRA")

    RetrieverClass = build_retriever_class()
    retriever = RetrieverClass(
        model_with_lora, tokenizer, sae, layer_idx=LAYER_IDX,
        topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
        target_norm=target_norm,
    )

    # Load IDF weights
    if os.path.exists(retrieval_ckpt):
        ckpt = torch.load(retrieval_ckpt, map_location="cpu", weights_only=False)
        if "idf_weights" in ckpt:
            retriever.idf_weights = ckpt["idf_weights"]
            log("IDF weights loaded from checkpoint")

    # --- Encode corpus ---
    log(f"Encoding {len(corpus_docs)} corpus documents...")
    batch_size = 16
    all_doc_vecs = []

    with torch.no_grad():
        for i in range(0, len(corpus_docs), batch_size):
            if i % 1000 == 0:
                log(f"  encoding docs: {i}/{len(corpus_docs)}")
            batch = corpus_docs[i:i + batch_size]
            d_sparse, _ = retriever.encode_documents(batch)
            all_doc_vecs.append(d_sparse.cpu())

    doc_matrix = torch.cat(all_doc_vecs, dim=0)  # (N_corpus, d_sae)
    log(f"Corpus encoded: {doc_matrix.shape}")

    # --- BM25 baseline index ---
    from rank_bm25 import BM25Okapi

    log("Building BM25 index...")
    tokenized_corpus = [doc.split() for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    log("BM25 index built")

    # --- Evaluate each detect task ---
    results_per_task = []
    lsr_all_precisions = []
    bm25_all_precisions = []
    hybrid_all_precisions = []
    lsr_coverage_hits = 0
    bm25_coverage_hits = 0
    hybrid_coverage_hits = 0
    total_findings = 0
    lsr_mrr_sum = 0.0
    bm25_mrr_sum = 0.0
    hybrid_mrr_sum = 0.0

    for task in all_tasks:
        audit_id = task["audit_id"]
        findings = task["findings"]
        log(f"\n--- {audit_id}: {len(findings)} findings, {task['n_sol_files']} sol files ---")

        task_results = {
            "audit_id": audit_id,
            "n_findings": len(findings),
            "findings_detail": [],
        }

        for finding in findings:
            # Build query from finding
            query = f"{finding['id']}: {finding['title']}"
            vuln_types = finding["vuln_types"]
            total_findings += 1

            # --- SCAR retrieval ---
            with torch.no_grad():
                q_sparse, _ = retriever.encode_queries([query])

            # Compute similarity to all corpus docs
            sim_scores = torch.matmul(q_sparse.cpu(), doc_matrix.T).squeeze(0)
            top_k_indices = sim_scores.argsort(descending=True)[:top_k_retrieve]
            top_k_scores = sim_scores[top_k_indices]

            # Check relevance: do retrieved docs contain similar vulnerability patterns?
            lsr_relevant = 0
            lsr_first_relevant_rank = None
            for rank, idx in enumerate(top_k_indices):
                doc = corpus_docs[idx.item()]
                doc_labels = corpus_labels[idx.item()] if idx.item() < len(corpus_labels) else ""
                doc_combined = f"{doc} {doc_labels}".lower()

                # Check if doc matches any of the finding's vulnerability types
                is_relevant = False
                for vtype in vuln_types:
                    if vtype == "other":
                        continue
                    for kw in VULN_TYPE_KEYWORDS.get(vtype, []):
                        if kw in doc_combined:
                            is_relevant = True
                            break
                    if is_relevant:
                        break

                if is_relevant:
                    lsr_relevant += 1
                    if lsr_first_relevant_rank is None:
                        lsr_first_relevant_rank = rank + 1

            lsr_precision = lsr_relevant / top_k_retrieve
            lsr_all_precisions.append(lsr_precision)
            if lsr_first_relevant_rank is not None:
                lsr_coverage_hits += 1
                lsr_mrr_sum += 1.0 / lsr_first_relevant_rank

            # --- BM25 retrieval ---
            q_tokens = query.split()
            bm25_scores = bm25.get_scores(q_tokens)
            bm25_top_k = bm25_scores.argsort()[::-1][:top_k_retrieve]

            bm25_relevant = 0
            bm25_first_relevant_rank = None
            for rank, idx in enumerate(bm25_top_k):
                doc = corpus_docs[idx]
                doc_labels = corpus_labels[idx] if idx < len(corpus_labels) else ""
                doc_combined = f"{doc} {doc_labels}".lower()

                is_relevant = False
                for vtype in vuln_types:
                    if vtype == "other":
                        continue
                    for kw in VULN_TYPE_KEYWORDS.get(vtype, []):
                        if kw in doc_combined:
                            is_relevant = True
                            break
                    if is_relevant:
                        break

                if is_relevant:
                    bm25_relevant += 1
                    if bm25_first_relevant_rank is None:
                        bm25_first_relevant_rank = rank + 1

            bm25_precision = bm25_relevant / top_k_retrieve
            bm25_all_precisions.append(bm25_precision)
            if bm25_first_relevant_rank is not None:
                bm25_coverage_hits += 1
                bm25_mrr_sum += 1.0 / bm25_first_relevant_rank

            # --- Hybrid LSR+BM25 retrieval (alpha=0.5) ---
            import numpy as np
            lsr_np = sim_scores.numpy()
            bm25_np = bm25_scores
            # Min-max normalize both
            lsr_min, lsr_max = lsr_np.min(), lsr_np.max()
            if lsr_max > lsr_min:
                lsr_norm = (lsr_np - lsr_min) / (lsr_max - lsr_min)
            else:
                lsr_norm = np.zeros_like(lsr_np)
            bm25_min, bm25_max = bm25_np.min(), bm25_np.max()
            if bm25_max > bm25_min:
                bm25_norm = (bm25_np - bm25_min) / (bm25_max - bm25_min)
            else:
                bm25_norm = np.zeros_like(bm25_np)
            hybrid_scores = 0.5 * lsr_norm + 0.5 * bm25_norm
            hybrid_top_k = hybrid_scores.argsort()[::-1][:top_k_retrieve]

            hybrid_relevant = 0
            hybrid_first_relevant_rank = None
            for rank, idx in enumerate(hybrid_top_k):
                doc = corpus_docs[idx]
                doc_labels = corpus_labels[idx] if idx < len(corpus_labels) else ""
                doc_combined = f"{doc} {doc_labels}".lower()
                is_relevant = False
                for vtype in vuln_types:
                    if vtype == "other":
                        continue
                    for kw in VULN_TYPE_KEYWORDS.get(vtype, []):
                        if kw in doc_combined:
                            is_relevant = True
                            break
                    if is_relevant:
                        break
                if is_relevant:
                    hybrid_relevant += 1
                    if hybrid_first_relevant_rank is None:
                        hybrid_first_relevant_rank = rank + 1

            hybrid_precision = hybrid_relevant / top_k_retrieve
            hybrid_all_precisions.append(hybrid_precision)
            if hybrid_first_relevant_rank is not None:
                hybrid_coverage_hits += 1
                hybrid_mrr_sum += 1.0 / hybrid_first_relevant_rank

            finding_result = {
                "finding_id": finding["id"],
                "title": finding["title"],
                "vuln_types": vuln_types,
                "lsr_precision_at_k": round(lsr_precision, 3),
                "lsr_first_relevant_rank": lsr_first_relevant_rank,
                "bm25_precision_at_k": round(bm25_precision, 3),
                "bm25_first_relevant_rank": bm25_first_relevant_rank,
                "hybrid_precision_at_k": round(hybrid_precision, 3),
                "hybrid_first_relevant_rank": hybrid_first_relevant_rank,
            }
            task_results["findings_detail"].append(finding_result)

            log(f"  {finding['id']}: types={vuln_types} | "
                f"LSR P@{top_k_retrieve}={lsr_precision:.2f} rank={lsr_first_relevant_rank} | "
                f"BM25 P@{top_k_retrieve}={bm25_precision:.2f} rank={bm25_first_relevant_rank} | "
                f"Hyb P@{top_k_retrieve}={hybrid_precision:.2f} rank={hybrid_first_relevant_rank}")

        results_per_task.append(task_results)

    # --- Aggregate metrics ---
    results = {
        "n_tasks": len(all_tasks),
        "n_findings": total_findings,
        "top_k": top_k_retrieve,
        "lsr": {
            "mean_precision_at_k": round(sum(lsr_all_precisions) / max(len(lsr_all_precisions), 1), 4),
            "coverage": round(lsr_coverage_hits / max(total_findings, 1), 4),
            "mrr": round(lsr_mrr_sum / max(total_findings, 1), 4),
        },
        "bm25": {
            "mean_precision_at_k": round(sum(bm25_all_precisions) / max(len(bm25_all_precisions), 1), 4),
            "coverage": round(bm25_coverage_hits / max(total_findings, 1), 4),
            "mrr": round(bm25_mrr_sum / max(total_findings, 1), 4),
        },
        "hybrid": {
            "mean_precision_at_k": round(sum(hybrid_all_precisions) / max(len(hybrid_all_precisions), 1), 4),
            "coverage": round(hybrid_coverage_hits / max(total_findings, 1), 4),
            "mrr": round(hybrid_mrr_sum / max(total_findings, 1), 4),
        },
        "per_task": results_per_task,
        "run_name": run_name,
        "sae_run_name": sae_run_name,
    }

    # --- Print summary ---
    log("\n" + "=" * 60)
    log("EVMbench Detect Mode — Retrieval Evaluation Summary")
    log("=" * 60)
    log(f"Tasks: {len(all_tasks)}, Findings: {total_findings}, Top-K: {top_k_retrieve}")
    log(f"")
    log(f"{'Method':<25} {'P@K':>8} {'Coverage':>10} {'MRR':>8}")
    log(f"-" * 55)
    log(f"{'BM25':<25} {results['bm25']['mean_precision_at_k']:>8.3f} "
        f"{results['bm25']['coverage']:>10.3f} {results['bm25']['mrr']:>8.3f}")
    log(f"{'SCAR':<25} {results['lsr']['mean_precision_at_k']:>8.3f} "
        f"{results['lsr']['coverage']:>10.3f} {results['lsr']['mrr']:>8.3f}")
    log(f"{'LSR+BM25 Hybrid':<25} {results['hybrid']['mean_precision_at_k']:>8.3f} "
        f"{results['hybrid']['coverage']:>10.3f} {results['hybrid']['mrr']:>8.3f}")
    log("=" * 60)

    # LSR vs BM25 deltas
    p_delta = results['lsr']['mean_precision_at_k'] - results['bm25']['mean_precision_at_k']
    c_delta = results['lsr']['coverage'] - results['bm25']['coverage']
    m_delta = results['lsr']['mrr'] - results['bm25']['mrr']
    log(f"LSR vs BM25: P@K {p_delta:+.3f}, Coverage {c_delta:+.3f}, MRR {m_delta:+.3f}")
    hp_delta = results['hybrid']['mean_precision_at_k'] - results['bm25']['mean_precision_at_k']
    hc_delta = results['hybrid']['coverage'] - results['bm25']['coverage']
    hm_delta = results['hybrid']['mrr'] - results['bm25']['mrr']
    log(f"Hybrid vs BM25: P@K {hp_delta:+.3f}, Coverage {hc_delta:+.3f}, MRR {hm_delta:+.3f}")

    # --- Save results ---
    results_dir = f"{SAVE_DIR}/evmbench"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/{run_name}_evmbench.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    retrieval_vol.commit()

    log(f"Results saved to {results_path}")

    # Push to HF
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        repo_id = f"{HF_USERNAME}/scar-weights"
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True,
                        token=hf_token)
        api.upload_file(
            path_or_fileobj=results_path,
            path_in_repo=f"evmbench/{run_name}_evmbench.json",
            repo_id=repo_id, repo_type="model", token=hf_token,
            commit_message=f"EVMbench results: {run_name}",
        )
        log(f"Results pushed to HF: {repo_id}")
    except Exception as e:
        log(f"HF push failed (non-fatal): {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(mode: str = "full", run_name: str = "primary",
         sae_run_name: str = "primary_L19_16k",
         sae_lora_rank: int = 0):
    """Run EVMbench evaluation.

    Args:
        mode: "full" | "setup"
        run_name: retrieval run name to evaluate
        sae_run_name: SAE checkpoint to use
        sae_lora_rank: SAE LoRA rank (0 = frozen SAE)
    """
    from datetime import datetime
    import json

    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"SCAR EVMbench Evaluation (mode={mode})")

    if mode == "setup":
        print("Setup mode — will only clone repos (done inside eval function)")
        return

    print(f"\n--- EVMbench Detect Mode (run={run_name}, sae={sae_run_name}, sae_lora_rank={sae_lora_rank}) ---")
    result = eval_evmbench.remote(
        run_name=run_name,
        sae_run_name=sae_run_name,
        sae_lora_rank=sae_lora_rank,
    )

    if result:
        print(f"\n=== RESULTS ===")
        print(f"Tasks: {result.get('n_tasks', 0)}, Findings: {result.get('n_findings', 0)}")
        for method in ["bm25", "lsr", "hybrid"]:
            if method in result:
                m = result[method]
                print(f"  {method.upper()}: P@K={m['mean_precision_at_k']:.3f}, "
                      f"Coverage={m['coverage']:.3f}, MRR={m['mrr']:.3f}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
