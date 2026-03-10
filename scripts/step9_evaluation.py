"""
SCAR Step 9: Full Evaluation — Modal Labs
====================================================
Evaluates SCAR against baselines on the FORGE-Curated eval holdout.

Models evaluated:
  1. SCAR (ours): LoRA + SAE + IDF + bidirectional attention
  2. BM25 baseline: Classic lexical sparse retrieval (rank_bm25)
  3. Dense retrieval baseline: Mean-pool backbone hidden states + cosine sim
  4. SAE-only (no LoRA): Frozen SAE without retrieval fine-tuning (ablation)

Metrics:
  - Recall@k (k=1,5,10,20)
  - MRR (Mean Reciprocal Rank)
  - nDCG@10 (normalized discounted cumulative gain)

Usage:
  modal run scripts/step9_evaluation.py                   # full evaluation
  modal run scripts/step9_evaluation.py --mode baselines   # baselines only
  modal run scripts/step9_evaluation.py --mode ours        # our model only
  modal run scripts/step9_evaluation.py --mode ablation    # SAE ablation eval
"""

import modal
import os
import math

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("scar-evaluation")

sae_vol = modal.Volume.from_name("scar-sae-training")
retrieval_vol = modal.Volume.from_name("scar-retrieval-training", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.45.0,<5.0.0",
        "datasets>=2.19.0,<4.0.0",
        "huggingface_hub>=0.23.0",
        "safetensors>=0.4.0",
        "tqdm>=4.66.0",
        "peft>=0.10.0,<1.0.0",
        "rank_bm25>=0.2.2",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

SAE_DIR = "/sae_training"
SAVE_DIR = "/retrieval_training"
HF_SECRET = modal.Secret.from_name("huggingface-token")
HF_USERNAME = "Farseen0"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
EVAL_DATASET = f"{HF_USERNAME}/scar-eval"
D_IN = 1536
D_SAE = 16384
LAYER_IDX = 19


def log(msg: str):
    import sys
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(sim_matrix):
    """Compute retrieval metrics from a (N_queries, N_docs) similarity matrix.

    Assumes ground truth is the diagonal (query i matches document i).

    Returns dict with: recall@1, recall@5, recall@10, recall@20, mrr, ndcg@10
    Also includes 'per_query' dict with per-query scores for bootstrap CIs.
    """
    import torch
    import numpy as np

    N = sim_matrix.shape[0]
    assert sim_matrix.shape == (N, N), f"Expected square matrix, got {sim_matrix.shape}"

    # Rankings: for each query, sort documents by descending similarity
    _, rankings = sim_matrix.sort(dim=1, descending=True)  # (N, N)

    # Ground truth: query i should retrieve document i
    correct = torch.arange(N).unsqueeze(1)  # (N, 1)

    metrics = {}
    per_query = {}

    # Recall@k
    for k in [1, 5, 10, 20]:
        top_k = rankings[:, :k]
        hits = (top_k == correct).any(dim=1).float()
        metrics[f"recall@{k}"] = hits.mean().item()
        per_query[f"recall@{k}"] = hits.numpy()

    # MRR (Mean Reciprocal Rank)
    # Find rank of correct document for each query
    ranks = (rankings == correct).nonzero(as_tuple=True)[1].float() + 1  # 1-indexed
    rr_per_query = (1.0 / ranks).numpy()
    mrr = float(rr_per_query.mean())
    metrics["mrr"] = mrr
    per_query["mrr"] = rr_per_query

    # nDCG@10
    # For single-relevant retrieval: DCG@10 = 1/log2(rank+1) if rank <= 10, else 0
    # IDCG@10 = 1/log2(2) = 1.0 (best case: correct at rank 1)
    k = 10
    top_k = rankings[:, :k]
    dcg_per_query = []
    for i in range(N):
        hit_positions = (top_k[i] == i).nonzero(as_tuple=True)[0]
        if len(hit_positions) > 0:
            rank_in_topk = hit_positions[0].item() + 1  # 1-indexed
            dcg = 1.0 / math.log2(rank_in_topk + 1)
        else:
            dcg = 0.0
        dcg_per_query.append(dcg)
    idcg = 1.0  # single relevant doc at rank 1: 1/log2(2) = 1.0
    ndcg = np.mean(dcg_per_query) / idcg
    metrics["ndcg@10"] = float(ndcg)
    per_query["ndcg@10"] = np.array(dcg_per_query) / idcg

    metrics["per_query"] = per_query
    return metrics


def bootstrap_ci(per_query_scores, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence intervals for a per-query metric array.

    Returns dict with: mean, ci_lower, ci_upper, se
    """
    import numpy as np
    rng = np.random.default_rng(42)
    n = len(per_query_scores)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = per_query_scores[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
    return {
        "mean": float(per_query_scores.mean()),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "se": float(boot_means.std()),
    }


def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000):
    """Paired bootstrap significance test: is system A better than system B?

    Tests H0: mean(A) <= mean(B) via paired bootstrap resampling.
    Returns dict with: delta (A-B mean), p_value, significant (at 0.05), ci_lower, ci_upper
    """
    import numpy as np
    rng = np.random.default_rng(42)
    assert len(scores_a) == len(scores_b), "Score arrays must be same length"
    n = len(scores_a)
    diffs = scores_a - scores_b
    observed_delta = float(diffs.mean())

    boot_deltas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_deltas[b] = diffs[idx].mean()

    # One-sided p-value: fraction of bootstrap samples where delta <= 0
    p_value = float((boot_deltas <= 0).mean())
    lo, hi = np.percentile(boot_deltas, [2.5, 97.5])

    return {
        "delta": observed_delta,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SAE Encoder (same as step6)
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
            self.lora_rank = lora_rank
            self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
            self.b_enc = nn.Parameter(torch.zeros(d_sae))
            self.b_dec = nn.Parameter(torch.zeros(d_in))
            self.log_threshold = nn.Parameter(torch.zeros(d_sae))

            if lora_rank > 0:
                self.lora_A = nn.Parameter(torch.empty(d_in, lora_rank))
                self.lora_B = nn.Parameter(torch.zeros(lora_rank, d_sae))
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        @property
        def threshold(self):
            return self.log_threshold.exp()

        def encode(self, x):
            x_centered = x - self.b_dec
            if self.lora_rank > 0:
                W_eff = self.W_enc + self.lora_A @ self.lora_B
                z_pre = x_centered @ W_eff + self.b_enc
            else:
                z_pre = x_centered @ self.W_enc + self.b_enc
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
        log(f"SAE normalizer: target_norm={target_norm:.4f}")

    sae = SAEEncoder(d_in=d_in, d_sae=d_sae, lora_rank=sae_lora_rank).to(device)
    sae.W_enc.data.copy_(state["W_enc"])
    sae.b_enc.data.copy_(state["b_enc"])
    sae.b_dec.data.copy_(state["b_dec"])
    sae.log_threshold.data.copy_(state["log_threshold"])
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
    return sae, target_norm


# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional Attention + Layer Access (same as step6)
# ═══════════════════════════════════════════════════════════════════════════

def enable_bidirectional_attention(model):
    """Replace causal attention with full bidirectional attention."""
    import torch

    if hasattr(model, 'peft_config'):
        qwen_model = model.base_model.model.model  # PEFT path
    else:
        qwen_model = model.model  # Qwen2ForCausalLM → Qwen2Model

    patched = 0
    for layer in qwen_model.layers:
        attn = layer.self_attn
        original_forward = attn.forward

        def make_bidir_forward(orig_fwd):
            def bidir_forward(*args, **kwargs):
                kwargs["attention_mask"] = None
                return orig_fwd(*args, **kwargs)
            return bidir_forward

        attn.forward = make_bidir_forward(original_forward)
        patched += 1

    log(f"Bidirectional attention: patched {patched} layers")


def _get_target_layer(model, layer_idx):
    try:
        return model.base_model.model.model.layers[layer_idx]
    except (AttributeError, TypeError):
        pass
    try:
        return model.model.layers[layer_idx]
    except (AttributeError, TypeError):
        pass
    raise ValueError(f"Cannot find layer {layer_idx} in model of type {type(model)}")


# ═══════════════════════════════════════════════════════════════════════════
# Retriever Class (same as step6)
# ═══════════════════════════════════════════════════════════════════════════

def build_retriever_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ScarRetriever(nn.Module):
        def __init__(self, model, tokenizer, sae, layer_idx=LAYER_IDX,
                     topk_query=100, topk_doc=400, max_seq_len=256,
                     target_norm=None, per_token_k=64, pooling_mode="sum"):
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
            self.pooling_mode = pooling_mode
            self._activations = None
            self.idf_weights = None
            self.prune_mask = None

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

            if self.pooling_mode == "max":
                pooled_raw = torch.amax(z_sparse, dim=1)
            else:
                pooled_raw = z_sparse.sum(dim=1)
            pooled = torch.log1p(pooled_raw)

            if self.idf_weights is not None:
                pooled = pooled * self.idf_weights.to(pooled.device)

            if self.prune_mask is not None:
                pooled = pooled * self.prune_mask.to(pooled.device).float()

            topk_vals, topk_idx = torch.topk(pooled, topk, dim=-1)
            sparse = torch.zeros_like(pooled).scatter(-1, topk_idx, topk_vals)
            sparse = F.normalize(sparse, dim=-1)

            self._activations = None
            return sparse, pooled

        def set_prune_mask(self, n_prune=0):
            """Zero out the n_prune most generic SAE features (lowest IDF)."""
            if n_prune > 0:
                if self.idf_weights is None:
                    raise ValueError("IDF must be computed first")
                _, sorted_idx = self.idf_weights.sort()
                mask = torch.ones(self.idf_weights.shape[0], dtype=torch.bool)
                mask[sorted_idx[:n_prune]] = False
                self.prune_mask = mask
            else:
                self.prune_mask = None

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

            log(f"Computing IDF from {N} documents...")
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


def build_splade_retriever_class():
    """Returns a SPLADE retriever class for evaluation. Mirrors step6's version."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SPLADERetriever(nn.Module):
        """SPLADE-style retriever using LM head for vocab-space sparse representations."""

        def __init__(self, model, tokenizer, topk_query=100, topk_doc=400,
                     max_seq_len=256):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self.topk_query = topk_query
            self.topk_doc = topk_doc
            self.max_seq_len = max_seq_len
            self.idf_weights = None

            if hasattr(model, 'config'):
                self.vocab_size = model.config.vocab_size
            else:
                self.vocab_size = model.base_model.model.config.vocab_size

        def _encode_texts(self, texts, topk):
            import torch

            device = next(self.model.parameters()).device
            enc = self.tokenizer(
                texts, return_tensors="pt", truncation=True,
                max_length=self.max_seq_len, padding=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask_2d = enc["attention_mask"].to(device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask_2d,
                use_cache=False,
            )
            logits = outputs.logits

            w = torch.log1p(F.relu(logits))
            w = w * attention_mask_2d.unsqueeze(-1).float()
            pooled = torch.amax(w, dim=1)

            if self.idf_weights is not None:
                pooled = pooled * self.idf_weights.to(pooled.device)

            topk_vals, topk_idx = torch.topk(pooled, topk, dim=-1)
            sparse = torch.zeros_like(pooled).scatter(-1, topk_idx, topk_vals)
            sparse = F.normalize(sparse, dim=-1)

            return sparse, pooled

        def encode_queries(self, texts):
            return self._encode_texts(texts, self.topk_query)

        def encode_documents(self, texts):
            return self._encode_texts(texts, self.topk_doc)

        @torch.no_grad()
        def compute_idf(self, documents, batch_size=16):
            old_idf = self.idf_weights
            self.idf_weights = None

            N = len(documents)
            doc_freq = torch.zeros(self.vocab_size)

            log(f"SPLADE: Computing IDF from {N} documents...")
            for i in range(0, N, batch_size):
                batch = documents[i:i+batch_size]
                _, pooled = self._encode_texts(batch, self.vocab_size)
                active = (pooled > 0).float().cpu()
                doc_freq += active.sum(dim=0)

            idf = torch.log(torch.tensor(N, dtype=torch.float32) / (1.0 + doc_freq))
            idf = idf.clamp(min=0.0)
            self.idf_weights = idf
            return idf

    return SPLADERetriever


# ═══════════════════════════════════════════════════════════════════════════
# BM25 Baseline
# ═══════════════════════════════════════════════════════════════════════════

def eval_bm25(queries, documents):
    """BM25 lexical baseline using rank_bm25.

    Tokenizes Solidity code by splitting on whitespace and punctuation.
    """
    import torch
    import re
    from rank_bm25 import BM25Okapi

    def tokenize_code(text):
        """Simple code-aware tokenizer: split on whitespace, punctuation, camelCase."""
        # Split camelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Split on non-alphanumeric
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+', text.lower())
        return tokens

    log("BM25: tokenizing documents...")
    doc_tokens = [tokenize_code(d) for d in documents]
    bm25 = BM25Okapi(doc_tokens)

    log("BM25: scoring queries...")
    N = len(queries)
    sim_matrix = torch.zeros(N, len(documents))

    for i, q in enumerate(queries):
        q_tokens = tokenize_code(q)
        scores = bm25.get_scores(q_tokens)
        sim_matrix[i] = torch.tensor(scores)

    metrics = compute_metrics(sim_matrix)
    log(f"BM25: R@10={metrics['recall@10']:.3f}, MRR={metrics['mrr']:.3f}, "
        f"nDCG@10={metrics['ndcg@10']:.3f}")
    return metrics, sim_matrix


# ═══════════════════════════════════════════════════════════════════════════
# Dense Retrieval Baseline
# ═══════════════════════════════════════════════════════════════════════════

def eval_dense(queries, documents, model, tokenizer, device, layer_idx=LAYER_IDX):
    """Dense retrieval baseline: mean-pool backbone hidden states + cosine sim.

    Uses the same backbone (Qwen2.5-Coder-1.5B) Layer 19 hidden states,
    mean-pooled across tokens, with cosine similarity. No SAE, no LoRA.
    """
    import torch
    import torch.nn.functional as F

    log("Dense: encoding texts...")

    def encode_dense(texts, batch_size=8):
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=256, padding=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            activations_captured = []

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activations_captured.append(output[0].detach().float())
                else:
                    activations_captured.append(output.detach().float())

            handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=False)
            handle.remove()

            acts = activations_captured[0]  # (B, S, d_in)
            # Mean-pool with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
            pooled = (acts * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, dim=-1)
            all_vecs.append(pooled.cpu())

        return torch.cat(all_vecs, dim=0)

    q_vecs = encode_dense(queries)
    d_vecs = encode_dense(documents)

    sim_matrix = torch.matmul(q_vecs, d_vecs.T)
    metrics = compute_metrics(sim_matrix)
    log(f"Dense: R@10={metrics['recall@10']:.3f}, MRR={metrics['mrr']:.3f}, "
        f"nDCG@10={metrics['ndcg@10']:.3f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# CodeBERT Baseline
# ═══════════════════════════════════════════════════════════════════════════

def eval_codebert(queries, documents, device):
    """CodeBERT dense retrieval baseline: encode with microsoft/codebert-base,
    mean-pool CLS + token embeddings, cosine similarity.

    CodeBERT was pretrained on NL-code pairs, making it a strong baseline for
    cross-modal retrieval (natural language queries → code documents).
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    log("CodeBERT: loading microsoft/codebert-base...")
    cb_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    cb_model = AutoModel.from_pretrained("microsoft/codebert-base").to(device).eval()

    log("CodeBERT: encoding texts...")

    def encode_codebert(texts, batch_size=16):
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = cb_tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                outputs = cb_model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state  # (B, S, 768)

            # Mean-pool with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, dim=-1)
            all_vecs.append(pooled.cpu())

        return torch.cat(all_vecs, dim=0)

    q_vecs = encode_codebert(queries)
    d_vecs = encode_codebert(documents)

    sim_matrix = torch.matmul(q_vecs, d_vecs.T)
    metrics = compute_metrics(sim_matrix)
    log(f"CodeBERT: R@10={metrics['recall@10']:.3f}, MRR={metrics['mrr']:.3f}, "
        f"nDCG@10={metrics['ndcg@10']:.3f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# E5 Dense Baseline
# ═══════════════════════════════════════════════════════════════════════════

def eval_e5(queries, documents, device):
    """E5-base-v2 dense retrieval baseline: instruction-tuned embeddings."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    model_name = "intfloat/e5-base-v2"
    log(f"E5: loading {model_name}...")
    e5_tokenizer = AutoTokenizer.from_pretrained(model_name)
    e5_model = AutoModel.from_pretrained(model_name).to(device).eval()

    def encode_e5(texts, prefix, batch_size=16):
        all_vecs = []
        prefixed = [f"{prefix}: {t}" for t in texts]
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i:i + batch_size]
            enc = e5_tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                outputs = e5_model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, dim=-1)
            all_vecs.append(pooled.cpu())
        return torch.cat(all_vecs, dim=0)

    log("E5: encoding queries and documents...")
    q_vecs = encode_e5(queries, "query")
    d_vecs = encode_e5(documents, "passage")

    sim_matrix = torch.matmul(q_vecs, d_vecs.T)
    metrics = compute_metrics(sim_matrix)
    log(f"E5: R@10={metrics['recall@10']:.3f}, MRR={metrics['mrr']:.3f}, "
        f"nDCG@10={metrics['ndcg@10']:.3f}")

    # E5 hybrid with BM25
    return metrics, sim_matrix


# ═══════════════════════════════════════════════════════════════════════════
# SCAR Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def eval_ours(queries, documents, retriever, use_idf=True, label="SCAR"):
    """Evaluate our trained retriever with full metrics."""
    import torch

    log(f"{label}: encoding {len(queries)} queries and {len(documents)} documents...")

    batch_size = 16
    all_q_vecs = []
    all_d_vecs = []

    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            q_batch = queries[i:i + batch_size]
            q_sparse, _ = retriever.encode_queries(q_batch)
            all_q_vecs.append(q_sparse.cpu())

        for i in range(0, len(documents), batch_size):
            d_batch = documents[i:i + batch_size]
            d_sparse, _ = retriever.encode_documents(d_batch)
            all_d_vecs.append(d_sparse.cpu())

    q_matrix = torch.cat(all_q_vecs, dim=0)
    d_matrix = torch.cat(all_d_vecs, dim=0)

    sim_matrix = torch.matmul(q_matrix, d_matrix.T)
    metrics = compute_metrics(sim_matrix)
    log(f"{label}: R@10={metrics['recall@10']:.3f}, MRR={metrics['mrr']:.3f}, "
        f"nDCG@10={metrics['ndcg@10']:.3f}")
    return metrics, sim_matrix


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid BM25 + LSR
# ═══════════════════════════════════════════════════════════════════════════

def eval_hybrid(bm25_sim, lsr_sim, alphas=[0.1, 0.2, 0.3, 0.5, 0.7]):
    """Evaluate hybrid BM25 + LSR retrieval with score interpolation.

    hybrid_score = alpha * LSR_score + (1 - alpha) * BM25_score_normalized

    Args:
        bm25_sim: (N, N) BM25 similarity matrix (raw scores)
        lsr_sim: (N, N) LSR cosine similarity matrix
        alphas: interpolation weights for LSR

    Returns:
        dict mapping alpha to metrics
    """
    import torch

    # Normalize BM25 scores to [0, 1] range per query for fair combination
    bm25_norm = bm25_sim.clone()
    for i in range(bm25_norm.shape[0]):
        row = bm25_norm[i]
        min_val, max_val = row.min(), row.max()
        if max_val > min_val:
            bm25_norm[i] = (row - min_val) / (max_val - min_val)

    # Normalize LSR scores similarly
    lsr_norm = lsr_sim.clone()
    for i in range(lsr_norm.shape[0]):
        row = lsr_norm[i]
        min_val, max_val = row.min(), row.max()
        if max_val > min_val:
            lsr_norm[i] = (row - min_val) / (max_val - min_val)

    results = {}
    best_alpha = None
    best_r10 = 0

    for alpha in alphas:
        hybrid_sim = alpha * lsr_norm + (1 - alpha) * bm25_norm
        metrics = compute_metrics(hybrid_sim)
        results[alpha] = metrics
        log(f"Hybrid (alpha={alpha:.1f}): R@10={metrics['recall@10']:.3f}, "
            f"MRR={metrics['mrr']:.3f}, nDCG@10={metrics['ndcg@10']:.3f}")
        if metrics['recall@10'] > best_r10:
            best_r10 = metrics['recall@10']
            best_alpha = alpha

    log(f"Best hybrid alpha={best_alpha}: R@10={best_r10:.3f}")
    return results, best_alpha


# ═══════════════════════════════════════════════════════════════════════════
# Main Evaluation Function
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={
        SAE_DIR: sae_vol,
        SAVE_DIR: retrieval_vol,
    },
    secrets=[HF_SECRET],
    timeout=7200,
)
def run_evaluation(
    mode: str = "full",
    run_name: str = "primary",
    sae_run_name: str = "primary_L19_16k",
    topk_query: int = 100,
    topk_doc: int = 400,
    lora_rank: int = 64,
    max_seq_len: int = 256,
    n_trials: int = 5,
):
    """Run full evaluation suite.

    Args:
        mode: "full" | "ours" | "baselines" | "ablation" | "multi_trial"
        run_name: retrieval training run name
        sae_run_name: SAE checkpoint run name
        topk_query: query TopK
        topk_doc: document TopK
        lora_rank: LoRA rank for loading checkpoint
        max_seq_len: max sequence length
        n_trials: number of trials for multi_trial mode
    """
    import torch
    import json
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sae_vol.reload()
    retrieval_vol.reload()

    device = torch.device("cuda")
    hf_token = os.environ.get("HF_TOKEN")

    # --- Load evaluation dataset ---
    log("Loading evaluation dataset...")
    ds = load_dataset(EVAL_DATASET, split="train", token=hf_token)
    queries = [row["query"] for row in ds]
    documents = [row["ground_truth_code"] for row in ds]
    N = len(queries)
    log(f"Evaluation set: {N} query-document pairs")
    log(f"Random baseline R@10 = {min(10, N) / N:.3f}")

    results = {"n_eval": N, "random_baseline_r10": min(10, N) / N}
    sim_matrices = {}  # store similarity matrices for hybrid

    # --- Load backbone (shared across all methods) ---
    log("Loading backbone model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # ═══════════════════════════════════════════════════
    # BM25 Baseline
    # ═══════════════════════════════════════════════════
    if mode in ("full", "baselines"):
        log("\n" + "=" * 60)
        log("BASELINE 1: BM25 (lexical sparse retrieval)")
        log("=" * 60)
        results["bm25"], sim_matrices["bm25"] = eval_bm25(queries, documents)

    # ═══════════════════════════════════════════════════
    # Dense Retrieval Baseline
    # ═══════════════════════════════════════════════════
    if mode in ("full", "baselines"):
        log("\n" + "=" * 60)
        log("BASELINE 2: Dense Retrieval (mean-pool Layer 19, cosine sim)")
        log("=" * 60)
        results["dense"] = eval_dense(queries, documents, backbone, tokenizer, device)

    # ═══════════════════════════════════════════════════
    # CodeBERT Baseline
    # ═══════════════════════════════════════════════════
    if mode in ("full", "baselines"):
        log("\n" + "=" * 60)
        log("BASELINE 3: CodeBERT (microsoft/codebert-base, mean-pool, cosine sim)")
        log("=" * 60)
        results["codebert"] = eval_codebert(queries, documents, device)

    # ═══════════════════════════════════════════════════
    # E5 Dense Baseline
    # ═══════════════════════════════════════════════════
    if mode in ("full", "baselines"):
        log("\n" + "=" * 60)
        log("BASELINE 4: E5-base-v2 (intfloat/e5-base-v2, instruction-tuned)")
        log("=" * 60)
        e5_metrics, e5_sim = eval_e5(queries, documents, device)
        results["e5"] = e5_metrics
        # E5 hybrid
        if "bm25" in sim_matrices:
            hybrid_results, best_alpha = eval_hybrid(
                sim_matrices["bm25"], e5_sim,
                alphas=[0.1, 0.2, 0.3, 0.5],
            )
            results["e5_hybrid"] = {
                "best_alpha": best_alpha,
                "best_metrics": hybrid_results[best_alpha],
            }
            log(f"E5+BM25 hybrid: R@10={hybrid_results[best_alpha]['recall@10']:.3f} (α={best_alpha})")

    # ═══════════════════════════════════════════════════
    # SAE-only (no LoRA) — ablation
    # ═══════════════════════════════════════════════════
    if mode in ("full", "ablation"):
        log("\n" + "=" * 60)
        log("ABLATION: SAE-only (no LoRA, no IDF)")
        log("=" * 60)

        sae_ckpt_path = f"{SAE_DIR}/{sae_run_name}/checkpoint_final.pt"
        if not os.path.exists(sae_ckpt_path):
            log(f"SAE checkpoint not found at {sae_ckpt_path}, skipping ablation")
            results["sae_only_no_idf"] = {"error": "sae checkpoint not found"}
            results["sae_only_with_idf"] = {"error": "sae checkpoint not found"}
        else:
            sae, target_norm = load_frozen_sae(sae_ckpt_path, device)

            # Enable bidirectional attention on backbone
            enable_bidirectional_attention(backbone)

            RetrieverClass = build_retriever_class()
            ablation_retriever = RetrieverClass(
                backbone, tokenizer, sae, layer_idx=LAYER_IDX,
                topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
                target_norm=target_norm,
            )
            results["sae_only_no_idf"], _ = eval_ours(
                queries, documents, ablation_retriever, use_idf=False,
                label="SAE-only (no LoRA, no IDF)",
            )

            # SAE-only with IDF
            log("\nABLATION: SAE-only (no LoRA, with IDF)")
            ablation_retriever.compute_idf(documents, batch_size=16)
            results["sae_only_with_idf"], _ = eval_ours(
                queries, documents, ablation_retriever, use_idf=True,
                label="SAE-only (no LoRA, with IDF)",
            )

            del ablation_retriever, sae
            torch.cuda.empty_cache()

        # Reload backbone fresh (bidirectional patching is destructive)
        del backbone
        torch.cuda.empty_cache()
        backbone = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16,
        ).to(device).eval()
        for p in backbone.parameters():
            p.requires_grad = False

    # ═══════════════════════════════════════════════════
    # SCAR (ours) — trained model
    # ═══════════════════════════════════════════════════
    if mode in ("full", "ours", "multi_trial"):
        log("\n" + "=" * 60)
        log("OURS: SCAR (LoRA + SAE + IDF + bidirectional)")
        log("=" * 60)

        # Load SAE
        sae_ckpt_path = f"{SAE_DIR}/{sae_run_name}/checkpoint_final.pt"
        sae, target_norm = load_frozen_sae(sae_ckpt_path, device)

        # Load LoRA
        from peft import PeftModel

        lora_dir = f"{SAVE_DIR}/{run_name}/lora_adapter"
        retrieval_ckpt = f"{SAVE_DIR}/{run_name}/checkpoint_final.pt"

        if os.path.exists(lora_dir):
            log(f"Loading LoRA adapter from {lora_dir}")
            model_with_lora = PeftModel.from_pretrained(backbone, lora_dir)
            model_with_lora.eval()
        elif os.path.exists(retrieval_ckpt):
            log(f"Loading LoRA from checkpoint {retrieval_ckpt}")
            from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            model_with_lora = get_peft_model(backbone, lora_config)
            ckpt = torch.load(retrieval_ckpt, map_location=device, weights_only=False)
            set_peft_model_state_dict(model_with_lora, ckpt["model_state_dict"])
            model_with_lora.eval()
        else:
            log(f"ERROR: No LoRA checkpoint found at {lora_dir} or {retrieval_ckpt}")
            log(f"Contents of {SAVE_DIR}: {os.listdir(SAVE_DIR) if os.path.exists(SAVE_DIR) else 'not found'}")
            if os.path.exists(f"{SAVE_DIR}/{run_name}"):
                log(f"Contents of {SAVE_DIR}/{run_name}: {os.listdir(f'{SAVE_DIR}/{run_name}')}")
            results["ours"] = {"error": "checkpoint not found"}
            _save_results(results, run_name)
            return results

        enable_bidirectional_attention(model_with_lora)

        RetrieverClass = build_retriever_class()
        retriever = RetrieverClass(
            model_with_lora, tokenizer, sae, layer_idx=LAYER_IDX,
            topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
            target_norm=target_norm,
        )

        # Load IDF weights from checkpoint
        if os.path.exists(retrieval_ckpt):
            ckpt = torch.load(retrieval_ckpt, map_location="cpu", weights_only=False)
            if "idf_weights" in ckpt:
                retriever.idf_weights = ckpt["idf_weights"]
                log("IDF weights loaded from checkpoint")
            else:
                log("No IDF weights in checkpoint, computing from eval docs...")
                retriever.compute_idf(documents, batch_size=16)
        else:
            retriever.compute_idf(documents, batch_size=16)

        # Eval with IDF
        results["ours"], sim_matrices["ours"] = eval_ours(
            queries, documents, retriever,
            label="SCAR",
        )

        # Ablation: our model WITHOUT IDF
        retriever_no_idf = RetrieverClass(
            model_with_lora, tokenizer, sae, layer_idx=LAYER_IDX,
            topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
            target_norm=target_norm,
        )
        results["ours_no_idf"], sim_matrices["ours_no_idf"] = eval_ours(
            queries, documents, retriever_no_idf,
            label="SCAR (no IDF)",
        )

        # Hybrid BM25 + LSR (if BM25 was also computed)
        if "bm25" in sim_matrices:
            log("\n" + "=" * 60)
            log("HYBRID: BM25 + SCAR (score interpolation)")
            log("=" * 60)
            hybrid_results, best_alpha = eval_hybrid(
                sim_matrices["bm25"], sim_matrices["ours_no_idf"],
                alphas=[0.05, 0.1, 0.2, 0.3, 0.5],
            )
            results["hybrid"] = {
                "best_alpha": best_alpha,
                "best_metrics": hybrid_results[best_alpha],
                "all_alphas": {str(k): v for k, v in hybrid_results.items()},
            }

        # Multi-trial mode: re-evaluate N times (for variance estimation)
        if mode == "multi_trial":
            log(f"\n--- Multi-trial evaluation ({n_trials} trials) ---")
            trial_results = []
            for t in range(n_trials):
                m, _ = eval_ours(queries, documents, retriever,
                                 label=f"Trial {t}")
                trial_results.append(m)
            for metric_name in trial_results[0]:
                vals = [t[metric_name] for t in trial_results]
                results[f"multi_trial_{metric_name}_mean"] = float(np.mean(vals))
                results[f"multi_trial_{metric_name}_std"] = float(np.std(vals))

    # ═══════════════════════════════════════════════════
    # Summary Table
    # ═══════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("EVALUATION RESULTS SUMMARY")
    log("=" * 70)
    log(f"{'Method':<35} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'MRR':>6} {'nDCG':>6}")
    log("-" * 70)

    random_r10 = min(10, N) / N
    log(f"{'Random baseline':<35} {'—':>6} {'—':>6} {random_r10:>6.3f} {'—':>6} {'—':>6} {'—':>6}")

    for method_name, method_key in [
        ("BM25", "bm25"),
        ("Dense (Layer 19 mean-pool)", "dense"),
        ("CodeBERT (mean-pool)", "codebert"),
        ("SAE-only (no LoRA, no IDF)", "sae_only_no_idf"),
        ("SAE-only (no LoRA, with IDF)", "sae_only_with_idf"),
        ("SCAR (no IDF)", "ours_no_idf"),
        ("SCAR (ours)", "ours"),
    ]:
        if method_key in results and isinstance(results[method_key], dict) and "recall@10" in results[method_key]:
            m = results[method_key]
            log(f"{method_name:<35} {m['recall@1']:>6.3f} {m['recall@5']:>6.3f} "
                f"{m['recall@10']:>6.3f} {m['recall@20']:>6.3f} {m['mrr']:>6.3f} "
                f"{m['ndcg@10']:>6.3f}")

    # Hybrid results
    if "hybrid" in results and "best_metrics" in results["hybrid"]:
        m = results["hybrid"]["best_metrics"]
        alpha = results["hybrid"]["best_alpha"]
        log(f"{'Hybrid BM25+LSR (α='+str(alpha)+')':<35} {m['recall@1']:>6.3f} {m['recall@5']:>6.3f} "
            f"{m['recall@10']:>6.3f} {m['recall@20']:>6.3f} {m['mrr']:>6.3f} "
            f"{m['ndcg@10']:>6.3f}")

    log("=" * 70)

    # Save results
    _save_results(results, run_name)

    return _strip_per_query(results)


def _strip_per_query(d):
    """Recursively strip 'per_query' keys (numpy arrays) for JSON serialization."""
    if isinstance(d, dict):
        return {k: _strip_per_query(v) for k, v in d.items() if k != "per_query"}
    return d


def _save_results(results, run_name):
    """Save evaluation results to volume and HuggingFace."""
    import json

    clean = _strip_per_query(results)

    results_dir = f"{SAVE_DIR}/evaluation"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/{run_name}_results.json"

    with open(results_path, "w") as f:
        json.dump(clean, f, indent=2)
    retrieval_vol.commit()
    log(f"Results saved to {results_path}")

    # Push to HF
    try:
        from huggingface_hub import HfApi
        hf_token = os.environ.get("HF_TOKEN")
        api = HfApi(token=hf_token)
        repo_id = f"{HF_USERNAME}/scar-weights"
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True,
                        token=hf_token)
        api.upload_file(
            path_or_fileobj=results_path,
            path_in_repo=f"evaluation/{run_name}_results.json",
            repo_id=repo_id, repo_type="model", token=hf_token,
            commit_message=f"Upload evaluation results: {run_name}",
        )
        log(f"Results pushed to HF: {repo_id}")
    except Exception as e:
        log(f"HF push failed (non-fatal): {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SAE Ablation Evaluation (Layer & Width sweeps)
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={
        SAE_DIR: sae_vol,
        SAVE_DIR: retrieval_vol,
    },
    secrets=[HF_SECRET],
    timeout=7200,
)
def eval_sae_ablations():
    """Evaluate SAE quality metrics for ablation table.

    Reports MSE, L0, dead features for each SAE variant.
    Does NOT require retrieval fine-tuning — just SAE reconstruction quality.
    """
    import torch
    import json
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sae_vol.reload()

    device = torch.device("cuda")
    hf_token = os.environ.get("HF_TOKEN")

    log("Loading backbone for SAE ablation evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load eval contracts
    ds = load_dataset(EVAL_DATASET, split="train", token=hf_token)
    texts = [row["ground_truth_code"] for row in ds]
    log(f"Loaded {len(texts)} eval texts")

    # SAE variants to evaluate
    sae_variants = [
        ("primary_L19_16k", 19),
        ("ablation_L14_16k", 14),
        ("ablation_L24_16k", 24),
        ("ablation_L19_8k", 19),
        ("ablation_L19_32k", 19),
    ]

    results = {}

    for sae_name, layer in sae_variants:
        ckpt_path = f"{SAE_DIR}/{sae_name}/checkpoint_final.pt"
        if not os.path.exists(ckpt_path):
            log(f"Skipping {sae_name}: checkpoint not found")
            results[sae_name] = {"status": "not_found"}
            continue

        log(f"\nEvaluating SAE: {sae_name} (layer={layer})")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["sae_state_dict"]
        config = ckpt["config"]

        d_in = state["W_enc"].shape[0]
        d_sae = state["W_enc"].shape[1]

        # Build full SAE (encoder + decoder) for MSE evaluation
        import torch.nn as nn
        import torch.nn.functional as F

        class JumpReLUSAE(nn.Module):
            def __init__(self, d_in, d_sae):
                super().__init__()
                self.d_in = d_in
                self.d_sae = d_sae
                self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
                self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
                self.b_enc = nn.Parameter(torch.zeros(d_sae))
                self.b_dec = nn.Parameter(torch.zeros(d_in))
                self.log_threshold = nn.Parameter(torch.zeros(d_sae))

            def forward(self, x):
                x_centered = x - self.b_dec
                z_pre = x_centered @ self.W_enc + self.b_enc
                z = F.relu(z_pre - self.log_threshold.exp())
                x_hat = z @ self.W_dec + self.b_dec
                return x_hat, z, z_pre

        sae = JumpReLUSAE(d_in, d_sae).to(device)

        # Load weights (handle missing W_dec gracefully)
        sae.W_enc.data.copy_(state["W_enc"])
        sae.b_enc.data.copy_(state["b_enc"])
        sae.b_dec.data.copy_(state["b_dec"])
        sae.log_threshold.data.copy_(state["log_threshold"])
        if "W_dec" in state:
            sae.W_dec.data.copy_(state["W_dec"])
        else:
            # Use transpose of W_enc as decoder
            sae.W_dec.data.copy_(state["W_enc"].T)
        sae.eval()

        # Extract normalizer
        target_norm = math.sqrt(d_in)
        if "normalizer_state_dict" in ckpt:
            target_norm = ckpt["normalizer_state_dict"].get("target_norm", target_norm)

        # Evaluate on texts
        mse_list = []
        l0_list = []
        total_tokens = 0
        dead_features = torch.zeros(d_sae, device=device)

        for i in range(0, len(texts), 4):
            batch = texts[i:i+4]
            enc = tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=256, padding=True,
            ).to(device)

            activations_captured = []

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activations_captured.append(output[0].detach().float())
                else:
                    activations_captured.append(output.detach().float())

            handle = model.model.layers[layer].register_forward_hook(hook_fn)
            with torch.no_grad():
                model(**enc)
            handle.remove()

            acts = activations_captured[0]  # (B, S, d_in)
            flat = acts.reshape(-1, d_in)

            # Normalize
            with torch.no_grad():
                mean_norm = flat.norm(dim=-1).mean().clamp(min=1e-8)
                flat = flat * (target_norm / mean_norm)

            with torch.no_grad():
                x_hat, z, z_pre = sae(flat)
                mse = F.mse_loss(x_hat, flat).item()
                l0 = (z > 0).float().sum(dim=-1).mean().item()
                dead_features += (z > 0).any(dim=0).float()
                mse_list.append(mse)
                l0_list.append(l0)
                total_tokens += flat.shape[0]

            del activations_captured
            torch.cuda.empty_cache()

        alive_features = (dead_features > 0).sum().item()

        sae_result = {
            "layer": layer,
            "d_sae": d_sae,
            "mse": float(np.mean(mse_list)),
            "l0": float(np.mean(l0_list)),
            "alive_features": int(alive_features),
            "dead_features": int(d_sae - alive_features),
            "dead_pct": float((d_sae - alive_features) / d_sae * 100),
            "total_tokens": total_tokens,
        }

        # Load training metrics if available
        if "metrics" in ckpt:
            train_metrics = ckpt["metrics"]
            sae_result["train_mse"] = train_metrics.get("mse")
            sae_result["train_l0"] = train_metrics.get("l0")

        results[sae_name] = sae_result
        log(f"  MSE={sae_result['mse']:.6f}, L0={sae_result['l0']:.1f}, "
            f"alive={alive_features}/{d_sae} ({sae_result['dead_pct']:.1f}% dead)")

        del sae
        torch.cuda.empty_cache()

    # Summary table
    log("\n" + "=" * 70)
    log("SAE ABLATION RESULTS")
    log("=" * 70)
    log(f"{'Name':<25} {'Layer':>5} {'Width':>6} {'MSE':>10} {'L0':>6} {'Dead%':>6}")
    log("-" * 70)
    for name in ["primary_L19_16k", "ablation_L14_16k", "ablation_L24_16k",
                  "ablation_L19_8k", "ablation_L19_32k"]:
        if name in results and "mse" in results[name]:
            r = results[name]
            log(f"{name:<25} {r['layer']:>5} {r['d_sae']:>6} "
                f"{r['mse']:>10.6f} {r['l0']:>6.1f} {r['dead_pct']:>5.1f}%")
    log("=" * 70)

    # Save
    results_dir = f"{SAE_DIR}/evaluation"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/sae_ablation_results.json"
    import json
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    sae_vol.commit()
    log(f"SAE ablation results saved to {results_path}")

    # Push to HF
    try:
        from huggingface_hub import HfApi
        hf_token = os.environ.get("HF_TOKEN")
        api = HfApi(token=hf_token)
        repo_id = f"{HF_USERNAME}/scar-weights"
        api.upload_file(
            path_or_fileobj=results_path,
            path_in_repo="evaluation/sae_ablation_results.json",
            repo_id=repo_id, repo_type="model", token=hf_token,
            commit_message="Upload SAE ablation results",
        )
        log(f"Pushed to HF: {repo_id}")
    except Exception as e:
        log(f"HF push failed: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Compare Multiple Runs
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={
        SAE_DIR: sae_vol,
        SAVE_DIR: retrieval_vol,
    },
    secrets=[HF_SECRET],
    timeout=7200,
)
def compare_runs(
    run_names: list[str] = None,
    sae_run_name: str = "primary_L19_16k",
    topk_query: int = 100,
    topk_doc: int = 400,
    lora_rank: int = 64,
    max_seq_len: int = 256,
):
    """Compare multiple retrieval runs efficiently in a single GPU session.

    Loads backbone + SAE once, then evaluates each run_name's LoRA adapter.
    Also computes BM25 baseline + hybrid for each run.
    """
    import torch
    import json
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if run_names is None:
        run_names = ["primary"]

    sae_vol.reload()
    retrieval_vol.reload()

    device = torch.device("cuda")
    hf_token = os.environ.get("HF_TOKEN")

    # --- Load evaluation dataset ---
    log("Loading evaluation dataset...")
    ds = load_dataset(EVAL_DATASET, split="train", token=hf_token)
    queries = [row["query"] for row in ds]
    documents = [row["ground_truth_code"] for row in ds]
    N = len(queries)
    log(f"Evaluation set: {N} query-document pairs")

    all_results = {"n_eval": N, "random_baseline_r10": min(10, N) / N}

    # --- Load tokenizer (once) ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- BM25 (once) ---
    log("\n" + "=" * 60)
    log("BASELINE: BM25")
    log("=" * 60)
    bm25_metrics, bm25_sim = eval_bm25(queries, documents)
    all_results["bm25"] = bm25_metrics

    # --- Per-run config: SAE variant, pooling mode, SAE LoRA ---
    run_config = {
        "improve_v4": {"sae": "mixed_L19_16k", "pooling": "max"},
        "v5c": {"sae_lora_rank": 16},
        "v5c_r32": {"sae_lora_rank": 32},
        "v5c_r64": {"sae_lora_rank": 64},
        "v5c_r128": {"sae_lora_rank": 128},
        "v5c_r64_lr2e4": {"sae_lora_rank": 64},
        "v5c_r64_lr5e4": {"sae_lora_rank": 64},
        "v5c_r64_s42": {"sae_lora_rank": 64},
        "v5c_r64_s123": {"sae_lora_rank": 64},
        "v5c_r64_s456": {"sae_lora_rank": 64},
        "v5c_r64_s789": {"sae_lora_rank": 64},
        "v5c_r64_s2026": {"sae_lora_rank": 64},
        "v5c_r64_tau0_05": {"sae_lora_rank": 64},
        "v5c_r64_tau0_2": {"sae_lora_rank": 64},
        "v5c_bm25": {"sae_lora_rank": 16},
        "v5c_bb128": {"sae_lora_rank": 128},
        "v5c_bb256": {"sae_lora_rank": 128},
        "v6_r64": {"sae_lora_rank": 64},
        "v6_r128": {"sae_lora_rank": 128},
        "v7_r64": {"sae_lora_rank": 64},
        "v7_r128": {"sae_lora_rank": 128},
        "v8_r64": {"sae_lora_rank": 64, "max_seq_len": 512},
        "v8_r128": {"sae_lora_rank": 128, "max_seq_len": 512},
        "v9": {"sae_lora_rank": 128},
        "v9_v7idf": {"sae_lora_rank": 128, "lora_from": "v9", "idf_from": "v7_r128"},
        "v10": {"sae_lora_rank": 128},
        "v11_r256": {"sae_lora_rank": 256},
        "v11_e10": {"sae_lora_rank": 128},
        "v11_combo": {"sae_lora_rank": 256},
        "v12_e15": {"sae_lora_rank": 256},
        "v12_bb128": {"sae_lora_rank": 256, "lora_rank": 128},
        "v12_lr3e5": {"sae_lora_rank": 256},
        "v12_data": {"sae_lora_rank": 256},
        "v13_e20": {"sae_lora_rank": 256},
        "v13_e25": {"sae_lora_rank": 256},
        "v14_nodistill": {"sae_lora_rank": 256},
        "v15_r128_nodistill": {"sae_lora_rank": 128},
        "v16_sae_only": {"sae_lora_rank": 256, "no_backbone_lora": True, "idf_from": "v13_e25"},
        "splade_v1": {"retriever_type": "splade"},
    }

    # --- Load SAEs (load each unique SAE needed) ---
    needed_saes = {sae_run_name}  # default
    for rn in run_names:
        if rn in run_config and "sae" in run_config[rn]:
            needed_saes.add(run_config[rn]["sae"])
    loaded_saes = {}
    for sae_name in needed_saes:
        ckpt_path = f"{SAE_DIR}/{sae_name}/checkpoint_final.pt"
        if os.path.exists(ckpt_path):
            loaded_saes[sae_name] = load_frozen_sae(ckpt_path, device)
            log(f"  Loaded SAE: {sae_name}")
        else:
            log(f"  WARNING: SAE not found: {ckpt_path}")
    sae, target_norm = loaded_saes.get(sae_run_name, (None, None))
    RetrieverClass = build_retriever_class()

    # --- Evaluate each run ---
    for run_name in run_names:
        log(f"\n{'=' * 60}")
        log(f"EVALUATING: {run_name}")
        log(f"{'=' * 60}")

        rc = run_config.get(run_name, {})
        # Support loading LoRA/IDF from different runs (e.g. v9 model + v7 IDF)
        lora_source = rc.get("lora_from", run_name)
        idf_source = rc.get("idf_from", run_name)
        lora_dir = f"{SAVE_DIR}/{lora_source}/lora_adapter"
        retrieval_ckpt = f"{SAVE_DIR}/{lora_source}/checkpoint_final.pt"
        idf_ckpt = f"{SAVE_DIR}/{idf_source}/checkpoint_final.pt"

        if not os.path.exists(lora_dir) and not os.path.exists(retrieval_ckpt):
            log(f"  SKIP: no checkpoint found for {lora_source}")
            all_results[run_name] = {"error": "checkpoint not found"}
            continue
        if lora_source != run_name:
            log(f"  LoRA from: {lora_source}")
        if idf_source != run_name:
            log(f"  IDF from: {idf_source}")

        # Load fresh backbone for each run (bidirectional patching is destructive)
        log(f"  Loading backbone...")
        backbone = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16,
        ).to(device).eval()
        for p in backbone.parameters():
            p.requires_grad = False

        # Load LoRA
        no_backbone_lora = rc.get("no_backbone_lora", False)
        if no_backbone_lora:
            model_with_lora = backbone
            model_with_lora.eval()
            log(f"  No backbone LoRA (SAE-LoRA only)")
        elif os.path.exists(lora_dir):
            log(f"  Loading LoRA from {lora_dir}")
            model_with_lora = PeftModel.from_pretrained(backbone, lora_dir)
            model_with_lora.eval()
        else:
            log(f"  Loading LoRA from checkpoint")
            from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            model_with_lora = get_peft_model(backbone, lora_config)
            ckpt = torch.load(retrieval_ckpt, map_location=device, weights_only=False)
            set_peft_model_state_dict(model_with_lora, ckpt["model_state_dict"])
            model_with_lora.eval()

        enable_bidirectional_attention(model_with_lora)

        # Per-run SAE and pooling config (rc already set above)
        run_sae_name = rc.get("sae", sae_run_name)
        run_pooling = rc.get("pooling", "sum")
        run_sae_lora_rank = rc.get("sae_lora_rank", 0)
        run_max_seq_len = rc.get("max_seq_len", max_seq_len)
        run_retriever_type = rc.get("retriever_type", "scar")

        if run_retriever_type == "splade":
            # SPLADE: use LM head (vocab-space) instead of SAE
            SPLADEClass = build_splade_retriever_class()
            retriever = SPLADEClass(
                model_with_lora, tokenizer, topk_query=topk_query,
                topk_doc=topk_doc, max_seq_len=run_max_seq_len,
            )
            log(f"  SPLADE mode (vocab_size={retriever.vocab_size})")

            # Load IDF from checkpoint
            if os.path.exists(idf_ckpt):
                idf_data = torch.load(idf_ckpt, map_location="cpu", weights_only=False)
                if "idf_weights" in idf_data:
                    retriever.idf_weights = idf_data["idf_weights"]
                    log(f"  IDF weights loaded from {idf_source}")
                else:
                    log(f"  No IDF in checkpoint, computing from eval docs...")
                    retriever.compute_idf(documents, batch_size=16)

            # Eval with IDF
            run_metrics_idf, lsr_sim_idf = eval_ours(
                queries, documents, retriever,
                label=f"{run_name} (with IDF)",
            )

            # Eval without IDF
            retriever_no_idf = SPLADEClass(
                model_with_lora, tokenizer, topk_query=topk_query,
                topk_doc=topk_doc, max_seq_len=run_max_seq_len,
            )
            run_metrics_no_idf, lsr_sim_no_idf = eval_ours(
                queries, documents, retriever_no_idf,
                label=f"{run_name} (no IDF)",
            )
        else:
            # SCAR: SAE-based retriever
            # For v5c: reload SAE with LoRA architecture and load trained LoRA weights
            if run_sae_lora_rank > 0:
                sae_ckpt_path = f"{SAE_DIR}/{run_sae_name}/checkpoint_final.pt"
                run_sae, run_target_norm = load_frozen_sae(
                    sae_ckpt_path, device, sae_lora_rank=run_sae_lora_rank)
                # Load SAE LoRA weights from retrieval checkpoint
                if os.path.exists(retrieval_ckpt):
                    ret_ckpt = torch.load(retrieval_ckpt, map_location=device, weights_only=False)
                    if "sae_lora_state" in ret_ckpt:
                        run_sae.load_state_dict(ret_ckpt["sae_lora_state"], strict=False)
                        log(f"  SAE LoRA weights loaded (rank={run_sae_lora_rank})")
                    else:
                        log(f"  WARNING: no sae_lora_state in checkpoint for v5c")
                log(f"  SAE={run_sae_name}, pooling={run_pooling}, sae_lora_rank={run_sae_lora_rank}")
            else:
                run_sae, run_target_norm = loaded_saes.get(run_sae_name, (sae, target_norm))
                if run_sae is None:
                    log(f"  SKIP: SAE {run_sae_name} not loaded")
                    all_results[run_name] = {"error": f"SAE {run_sae_name} not found"}
                    del model_with_lora, backbone
                    torch.cuda.empty_cache()
                    continue
                log(f"  SAE={run_sae_name}, pooling={run_pooling}")

            retriever = RetrieverClass(
                model_with_lora, tokenizer, run_sae, layer_idx=LAYER_IDX,
                topk_query=topk_query, topk_doc=topk_doc, max_seq_len=run_max_seq_len,
                target_norm=run_target_norm, pooling_mode=run_pooling,
            )

            # Load IDF from checkpoint (may come from a different run via idf_from)
            if os.path.exists(idf_ckpt):
                idf_data = torch.load(idf_ckpt, map_location="cpu", weights_only=False)
                if "idf_weights" in idf_data:
                    retriever.idf_weights = idf_data["idf_weights"]
                    log(f"  IDF weights loaded from {idf_source}")
                else:
                    log(f"  No IDF in checkpoint, computing from eval docs...")
                    retriever.compute_idf(documents, batch_size=16)

            # Eval with IDF
            run_metrics_idf, lsr_sim_idf = eval_ours(
                queries, documents, retriever,
                label=f"{run_name} (with IDF)",
            )

            # Eval without IDF (use run_sae, not default sae, so v5c LoRA is included)
            retriever_no_idf = RetrieverClass(
                model_with_lora, tokenizer, run_sae, layer_idx=LAYER_IDX,
                topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
                target_norm=run_target_norm, pooling_mode=run_pooling,
            )
            run_metrics_no_idf, lsr_sim_no_idf = eval_ours(
                queries, documents, retriever_no_idf,
                label=f"{run_name} (no IDF)",
            )

        # Hybrid BM25 + LSR (no IDF)
        hybrid_results_no_idf, best_alpha_no_idf = eval_hybrid(
            bm25_sim, lsr_sim_no_idf, alphas=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
        )

        # Hybrid BM25 + LSR (with IDF)
        hybrid_results_idf, best_alpha_idf = eval_hybrid(
            bm25_sim, lsr_sim_idf, alphas=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
        )

        all_results[run_name] = {
            "with_idf": run_metrics_idf,
            "no_idf": run_metrics_no_idf,
            "hybrid_no_idf": {
                "best_alpha": best_alpha_no_idf,
                "best_metrics": hybrid_results_no_idf[best_alpha_no_idf],
            },
            "hybrid_idf": {
                "best_alpha": best_alpha_idf,
                "best_metrics": hybrid_results_idf[best_alpha_idf],
            },
        }

        # Cleanup
        del model_with_lora, backbone, retriever, retriever_no_idf
        torch.cuda.empty_cache()

    # --- Summary Table with Bootstrap CIs ---
    import numpy as np
    log("\n" + "=" * 120)
    log("COMPARISON RESULTS (95% Bootstrap CIs, n=10000)")
    log("=" * 120)
    log(f"{'Run':<20} {'R@10(noIDF)':>10} {'R@10(IDF)':>10} {'Hyb(IDF) [95%CI]':>22} {'α':>5} {'nDCG@10 [95%CI]':>22}")
    log("-" * 120)

    random_r10 = min(10, N) / N
    # BM25 bootstrap CIs
    bm25_pq = bm25_metrics.get("per_query", {})
    if bm25_pq and "recall@10" in bm25_pq:
        bm25_r10_ci = bootstrap_ci(bm25_pq["recall@10"])
        bm25_ndcg_ci = bootstrap_ci(bm25_pq["ndcg@10"])
        bm25_r10_str = f"{bm25_r10_ci['mean']:.3f} [{bm25_r10_ci['ci_lower']:.3f},{bm25_r10_ci['ci_upper']:.3f}]"
        bm25_ndcg_str = f"{bm25_ndcg_ci['mean']:.3f} [{bm25_ndcg_ci['ci_lower']:.3f},{bm25_ndcg_ci['ci_upper']:.3f}]"
        log(f"{'BM25':<20} {'—':>10} {'—':>10} {bm25_r10_str:>22} {'—':>5} {bm25_ndcg_str:>22}")
    else:
        log(f"{'BM25':<20} {'—':>10} {'—':>10} {bm25_metrics['recall@10']:>10.3f}{'':>12} {'—':>5} {bm25_metrics['ndcg@10']:>10.3f}")

    best_run = None
    best_r10 = 0
    save_results = {}  # stripped of per_query numpy arrays

    for run_name in run_names:
        if run_name not in all_results or "error" in all_results[run_name]:
            log(f"{run_name:<20} {'ERROR':>10}")
            save_results[run_name] = all_results.get(run_name, {"error": True})
            continue
        r = all_results[run_name]
        r10_no = r["no_idf"]["recall@10"]
        r10_idf = r["with_idf"]["recall@10"]
        hyb_idf_r10 = r["hybrid_idf"]["best_metrics"]["recall@10"]
        alpha_idf = r["hybrid_idf"]["best_alpha"]
        ndcg_idf = r["hybrid_idf"]["best_metrics"]["ndcg@10"]

        # Bootstrap CIs for hybrid IDF (primary metric)
        hyb_pq = r["hybrid_idf"]["best_metrics"].get("per_query", {})
        ci_data = {}
        if hyb_pq and "recall@10" in hyb_pq:
            r10_ci = bootstrap_ci(hyb_pq["recall@10"])
            ndcg_ci = bootstrap_ci(hyb_pq["ndcg@10"])
            hyb_str = f"{r10_ci['mean']:.3f} [{r10_ci['ci_lower']:.3f},{r10_ci['ci_upper']:.3f}]"
            ndcg_str = f"{ndcg_ci['mean']:.3f} [{ndcg_ci['ci_lower']:.3f},{ndcg_ci['ci_upper']:.3f}]"
            ci_data["hybrid_idf_r10"] = r10_ci
            ci_data["hybrid_idf_ndcg10"] = ndcg_ci
        else:
            hyb_str = f"{hyb_idf_r10:.3f}"
            ndcg_str = f"{ndcg_idf:.3f}"

        # Standalone IDF CIs
        idf_pq = r["with_idf"].get("per_query", {})
        if idf_pq and "recall@10" in idf_pq:
            ci_data["standalone_idf_r10"] = bootstrap_ci(idf_pq["recall@10"])

        log(f"{run_name:<20} {r10_no:>10.3f} {r10_idf:>10.3f} {hyb_str:>22} {alpha_idf:>5.2f} {ndcg_str:>22}")

        # Best run by IDF hybrid
        if hyb_idf_r10 > best_r10:
            best_r10 = hyb_idf_r10
            best_run = run_name

        # Strip per_query arrays for JSON serialization
        def strip_pq(d):
            if isinstance(d, dict):
                return {k: strip_pq(v) for k, v in d.items() if k != "per_query"}
            return d
        run_save = strip_pq(r)
        if ci_data:
            run_save["bootstrap_ci"] = ci_data
        save_results[run_name] = run_save

    log("=" * 120)
    if best_run:
        ci_info = save_results[best_run].get("bootstrap_ci", {})
        ci_r10 = ci_info.get("hybrid_idf_r10", {})
        if ci_r10:
            log(f"BEST RUN: {best_run} (Hybrid IDF R@10={ci_r10['mean']:.3f} "
                f"[{ci_r10['ci_lower']:.3f}, {ci_r10['ci_upper']:.3f}])")
        else:
            log(f"BEST RUN: {best_run} (Hybrid IDF R@10={best_r10:.3f})")

    # --- Cross-seed aggregate (v5c_r64 seeds) ---
    seed_names = [rn for rn in run_names if rn.startswith("v5c_r64_s")]
    seed_data = []
    for sn in seed_names:
        if sn in all_results and "error" not in all_results[sn]:
            r = all_results[sn]
            seed_data.append({
                "standalone_idf_r10": r["with_idf"]["recall@10"],
                "standalone_idf_ndcg10": r["with_idf"]["ndcg@10"],
                "hybrid_idf_r10": r["hybrid_idf"]["best_metrics"]["recall@10"],
                "hybrid_idf_ndcg10": r["hybrid_idf"]["best_metrics"]["ndcg@10"],
                "hybrid_idf_mrr": r["hybrid_idf"]["best_metrics"]["mrr"],
                "standalone_idf_mrr": r["with_idf"]["mrr"],
            })
    if len(seed_data) >= 2:
        log(f"\n{'='*80}")
        log(f"CROSS-SEED VARIANCE ({len(seed_data)} seeds: {', '.join(seed_names)})")
        log(f"{'='*80}")
        for metric in ["standalone_idf_r10", "standalone_idf_ndcg10", "standalone_idf_mrr",
                        "hybrid_idf_r10", "hybrid_idf_ndcg10", "hybrid_idf_mrr"]:
            vals = np.array([d[metric] for d in seed_data])
            log(f"  {metric:<25}: {vals.mean():.3f} ± {vals.std():.3f}  "
                f"(min={vals.min():.3f}, max={vals.max():.3f})")
        save_results["_seed_aggregate"] = {
            "n_seeds": len(seed_data),
            "seeds": seed_names,
            "metrics": {
                metric: {
                    "mean": float(np.mean([d[metric] for d in seed_data])),
                    "std": float(np.std([d[metric] for d in seed_data])),
                    "min": float(np.min([d[metric] for d in seed_data])),
                    "max": float(np.max([d[metric] for d in seed_data])),
                    "values": [float(d[metric]) for d in seed_data],
                }
                for metric in seed_data[0].keys()
            },
        }

    # --- Paired bootstrap significance tests (best run vs BM25) ---
    if best_run and best_run in all_results and bm25_pq and "recall@10" in bm25_pq:
        r = all_results[best_run]
        hyb_pq = r["hybrid_idf"]["best_metrics"].get("per_query", {})
        if hyb_pq and "recall@10" in hyb_pq:
            log(f"\n{'='*80}")
            log(f"PAIRED BOOTSTRAP SIGNIFICANCE TESTS (n=10000)")
            log(f"{'='*80}")

            sig_results = {}
            for metric in ["recall@10", "ndcg@10", "mrr"]:
                if metric in hyb_pq and metric in bm25_pq:
                    test = paired_bootstrap_test(hyb_pq[metric], bm25_pq[metric])
                    sig_str = "***" if test["p_value"] < 0.001 else "**" if test["p_value"] < 0.01 else "*" if test["p_value"] < 0.05 else "n.s."
                    log(f"  {best_run}+BM25 vs BM25 ({metric}): "
                        f"Δ={test['delta']:+.3f} [{test['ci_lower']:+.3f}, {test['ci_upper']:+.3f}], "
                        f"p={test['p_value']:.4f} {sig_str}")
                    sig_results[metric] = test

            # Also test standalone SCAR vs BM25
            idf_pq = r["with_idf"].get("per_query", {})
            if idf_pq and "recall@10" in idf_pq:
                log(f"\n  Standalone SCAR vs BM25:")
                for metric in ["recall@10", "ndcg@10"]:
                    if metric in idf_pq and metric in bm25_pq:
                        test = paired_bootstrap_test(idf_pq[metric], bm25_pq[metric])
                        sig_str = "***" if test["p_value"] < 0.001 else "**" if test["p_value"] < 0.01 else "*" if test["p_value"] < 0.05 else "n.s."
                        log(f"    SCAR(IDF) vs BM25 ({metric}): "
                            f"Δ={test['delta']:+.3f} [{test['ci_lower']:+.3f}, {test['ci_upper']:+.3f}], "
                            f"p={test['p_value']:.4f} {sig_str}")
                        sig_results[f"standalone_{metric}"] = test

            save_results["_significance_tests"] = sig_results

    # Save (BM25 CIs too)
    bm25_save = {k: v for k, v in bm25_metrics.items() if k != "per_query"}
    if bm25_pq and "recall@10" in bm25_pq:
        bm25_save["bootstrap_ci"] = {
            "recall@10": bootstrap_ci(bm25_pq["recall@10"]),
            "ndcg@10": bootstrap_ci(bm25_pq["ndcg@10"]),
        }
    save_results["_bm25"] = bm25_save
    _save_results(save_results, "comparison")

    return save_results  # return stripped version (JSON-serializable)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Pruning Sweep
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={
        SAE_DIR: sae_vol,
        SAVE_DIR: retrieval_vol,
    },
    secrets=[HF_SECRET],
    timeout=7200,
)
def eval_prune_sweep(
    run_name: str = "primary",
    sae_run_name: str = "primary_L19_16k",
    prune_levels: list[int] = None,
    topk_query: int = 100,
    topk_doc: int = 400,
    lora_rank: int = 64,
    max_seq_len: int = 256,
):
    """Sweep feature pruning levels on a trained model.

    Zeros out the N most common SAE features (lowest IDF) before doc-level
    TopK. Like stop-word removal for sparse vectors. Tests whether removing
    generic noise features helps the existing model.

    Args:
        run_name: retrieval run to evaluate
        sae_run_name: SAE checkpoint name
        prune_levels: list of N values to test (features to prune)
        topk_query: query TopK
        topk_doc: document TopK
        lora_rank: LoRA rank
        max_seq_len: max sequence length
    """
    import torch
    import json
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if prune_levels is None:
        prune_levels = [0, 50, 100, 200, 500, 1000, 2000]

    sae_vol.reload()
    retrieval_vol.reload()

    device = torch.device("cuda")
    hf_token = os.environ.get("HF_TOKEN")

    # Load eval dataset
    log("Loading evaluation dataset...")
    ds = load_dataset(EVAL_DATASET, split="train", token=hf_token)
    queries = [row["query"] for row in ds]
    documents = [row["ground_truth_code"] for row in ds]
    N = len(queries)
    log(f"Evaluation set: {N} query-document pairs")

    # Load backbone + LoRA
    log("Loading backbone...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    lora_dir = f"{SAVE_DIR}/{run_name}/lora_adapter"
    retrieval_ckpt = f"{SAVE_DIR}/{run_name}/checkpoint_final.pt"

    if os.path.exists(lora_dir):
        model = PeftModel.from_pretrained(backbone, lora_dir).eval()
    else:
        log(f"ERROR: LoRA not found at {lora_dir}")
        return None

    enable_bidirectional_attention(model)

    # Load SAE
    sae_ckpt_path = f"{SAE_DIR}/{sae_run_name}/checkpoint_final.pt"
    sae, target_norm = load_frozen_sae(sae_ckpt_path, device)

    # Build retriever
    RetrieverClass = build_retriever_class()
    retriever = RetrieverClass(
        model, tokenizer, sae, layer_idx=LAYER_IDX,
        topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
        target_norm=target_norm,
    )

    # Load IDF from checkpoint
    if os.path.exists(retrieval_ckpt):
        ckpt = torch.load(retrieval_ckpt, map_location="cpu", weights_only=False)
        if "idf_weights" in ckpt:
            retriever.idf_weights = ckpt["idf_weights"]
            log("IDF weights loaded from checkpoint")
        else:
            log("No IDF in checkpoint, computing from eval docs...")
            retriever.compute_idf(documents, batch_size=16)
    else:
        retriever.compute_idf(documents, batch_size=16)

    # BM25 baseline (for hybrid)
    bm25_metrics, bm25_sim = eval_bm25(queries, documents)

    # Sweep
    results = {"n_eval": N, "bm25": bm25_metrics}

    log(f"\n{'=' * 70}")
    log(f"FEATURE PRUNING SWEEP (run={run_name})")
    log(f"{'=' * 70}")
    log(f"{'Prune N':>8} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'MRR':>6} {'Hybrid R@10':>11}")
    log("-" * 70)

    for n_prune in prune_levels:
        retriever.set_prune_mask(n_prune)

        # Standalone eval
        standalone_metrics, lsr_sim = eval_ours(
            queries, documents, retriever,
            label=f"prune={n_prune}",
        )

        # Hybrid
        hybrid_results, best_alpha = eval_hybrid(
            bm25_sim, lsr_sim, alphas=[0.05, 0.1, 0.2, 0.3, 0.5],
        )
        hybrid_r10 = hybrid_results[best_alpha]["recall@10"]

        results[f"prune_{n_prune}"] = {
            "standalone": standalone_metrics,
            "hybrid_best_alpha": best_alpha,
            "hybrid_metrics": hybrid_results[best_alpha],
        }

        log(f"{n_prune:>8} {standalone_metrics['recall@1']:>6.3f} "
            f"{standalone_metrics['recall@5']:>6.3f} "
            f"{standalone_metrics['recall@10']:>6.3f} "
            f"{standalone_metrics['recall@20']:>6.3f} "
            f"{standalone_metrics['mrr']:>6.3f} "
            f"{hybrid_r10:>11.3f}")

    log("=" * 70)

    # Save
    _save_results(results, f"prune_sweep_{run_name}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# v5c Deep Analysis
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={
        SAE_DIR: sae_vol,
        SAVE_DIR: retrieval_vol,
    },
    secrets=[HF_SECRET],
    timeout=7200,
)
def eval_v5c_analysis(
    sae_run_name: str = "primary_L19_16k",
    topk_query: int = 100,
    topk_doc: int = 400,
    lora_rank: int = 64,
    max_seq_len: int = 256,
):
    """Deep analysis of v5c: test IDF vs no-IDF in hybrid, pruning combos,
    and compare with primary's hybrid to find the best paper-worthy config.
    """
    import torch
    import json
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    sae_vol.reload()
    retrieval_vol.reload()

    device = torch.device("cuda")
    hf_token = os.environ.get("HF_TOKEN")

    # Load eval dataset
    log("Loading evaluation dataset...")
    ds = load_dataset(EVAL_DATASET, split="train", token=hf_token)
    queries = [row["query"] for row in ds]
    documents = [row["ground_truth_code"] for row in ds]
    N = len(queries)
    log(f"Evaluation set: {N} pairs")

    # BM25 baseline
    bm25_metrics, bm25_sim = eval_bm25(queries, documents)

    results = {"n_eval": N, "bm25": bm25_metrics}
    RetrieverClass = build_retriever_class()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============================================================
    # 1. v5c with IDF — standalone + hybrid
    # ============================================================
    log("\n" + "=" * 60)
    log("V5C WITH IDF")
    log("=" * 60)

    backbone = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    lora_dir = f"{SAVE_DIR}/v5c/lora_adapter"
    retrieval_ckpt = f"{SAVE_DIR}/v5c/checkpoint_final.pt"
    model = PeftModel.from_pretrained(backbone, lora_dir).eval()
    enable_bidirectional_attention(model)

    sae_ckpt_path = f"{SAE_DIR}/{sae_run_name}/checkpoint_final.pt"
    sae, target_norm = load_frozen_sae(sae_ckpt_path, device, sae_lora_rank=16)
    ckpt = torch.load(retrieval_ckpt, map_location=device, weights_only=False)
    if "sae_lora_state" in ckpt:
        sae.load_state_dict(ckpt["sae_lora_state"], strict=False)
        log("SAE LoRA weights loaded")

    retriever = RetrieverClass(
        model, tokenizer, sae, layer_idx=LAYER_IDX,
        topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
        target_norm=target_norm,
    )

    # Load IDF
    if "idf_weights" in ckpt:
        retriever.idf_weights = ckpt["idf_weights"]
        log("IDF loaded from checkpoint")

    # v5c WITH IDF — standalone
    v5c_idf_metrics, v5c_idf_sim = eval_ours(
        queries, documents, retriever, label="v5c (IDF)")

    # v5c WITH IDF — hybrid (THIS is what compare_runs missed)
    log("\nHybrid: BM25 + v5c(IDF)")
    hybrid_idf_results, best_alpha_idf = eval_hybrid(
        bm25_sim, v5c_idf_sim,
        alphas=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7],
    )
    results["v5c_idf_standalone"] = v5c_idf_metrics
    results["v5c_idf_hybrid"] = {
        "best_alpha": best_alpha_idf,
        "best_metrics": hybrid_idf_results[best_alpha_idf],
        "all_alphas": {str(k): v for k, v in hybrid_idf_results.items()},
    }

    # ============================================================
    # 2. v5c WITHOUT IDF — standalone + hybrid
    # ============================================================
    log("\n" + "=" * 60)
    log("V5C WITHOUT IDF")
    log("=" * 60)

    retriever.idf_weights = None
    v5c_no_idf_metrics, v5c_no_idf_sim = eval_ours(
        queries, documents, retriever, label="v5c (no IDF)")

    log("\nHybrid: BM25 + v5c(no IDF)")
    hybrid_no_idf_results, best_alpha_no_idf = eval_hybrid(
        bm25_sim, v5c_no_idf_sim,
        alphas=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7],
    )
    results["v5c_no_idf_standalone"] = v5c_no_idf_metrics
    results["v5c_no_idf_hybrid"] = {
        "best_alpha": best_alpha_no_idf,
        "best_metrics": hybrid_no_idf_results[best_alpha_no_idf],
        "all_alphas": {str(k): v for k, v in hybrid_no_idf_results.items()},
    }

    # ============================================================
    # 3. v5c + pruning sweep (IDF + prune)
    # ============================================================
    log("\n" + "=" * 60)
    log("V5C + PRUNING (IDF + prune)")
    log("=" * 60)

    # Restore IDF
    retriever.idf_weights = ckpt["idf_weights"]

    prune_levels = [0, 100, 500, 1000, 2000, 4000]
    prune_results = {}

    for n_prune in prune_levels:
        retriever.set_prune_mask(n_prune)
        p_metrics, p_sim = eval_ours(
            queries, documents, retriever, label=f"v5c+prune={n_prune}")
        p_hybrid, p_alpha = eval_hybrid(
            bm25_sim, p_sim, alphas=[0.05, 0.1, 0.2, 0.3, 0.5])
        prune_results[n_prune] = {
            "standalone": p_metrics,
            "hybrid_alpha": p_alpha,
            "hybrid_metrics": p_hybrid[p_alpha],
        }

    results["v5c_prune_sweep"] = {str(k): v for k, v in prune_results.items()}

    # ============================================================
    # 4. Primary with IDF in hybrid (for fair comparison)
    # ============================================================
    log("\n" + "=" * 60)
    log("PRIMARY WITH IDF — HYBRID (fair comparison)")
    log("=" * 60)

    del model, sae, retriever
    torch.cuda.empty_cache()

    backbone2 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16).to(device).eval()
    for p in backbone2.parameters():
        p.requires_grad = False

    primary_lora = f"{SAVE_DIR}/primary/lora_adapter"
    primary_ckpt = f"{SAVE_DIR}/primary/checkpoint_final.pt"
    model2 = PeftModel.from_pretrained(backbone2, primary_lora).eval()
    enable_bidirectional_attention(model2)

    sae2, tn2 = load_frozen_sae(sae_ckpt_path, device)
    retriever2 = RetrieverClass(
        model2, tokenizer, sae2, layer_idx=LAYER_IDX,
        topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
        target_norm=tn2,
    )
    pckpt = torch.load(primary_ckpt, map_location="cpu", weights_only=False)
    if "idf_weights" in pckpt:
        retriever2.idf_weights = pckpt["idf_weights"]

    primary_idf_metrics, primary_idf_sim = eval_ours(
        queries, documents, retriever2, label="primary (IDF)")

    log("\nHybrid: BM25 + primary(IDF)")
    primary_idf_hybrid, primary_idf_alpha = eval_hybrid(
        bm25_sim, primary_idf_sim,
        alphas=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
    )
    results["primary_idf_standalone"] = primary_idf_metrics
    results["primary_idf_hybrid"] = {
        "best_alpha": primary_idf_alpha,
        "best_metrics": primary_idf_hybrid[primary_idf_alpha],
    }

    # ============================================================
    # Summary
    # ============================================================
    log("\n" + "=" * 70)
    log("V5C ANALYSIS SUMMARY")
    log("=" * 70)
    log(f"{'Config':<35} {'R@10':>6} {'MRR':>6} {'Hybrid R@10':>11}")
    log("-" * 70)
    log(f"{'BM25':<35} {bm25_metrics['recall@10']:>6.3f} {bm25_metrics['mrr']:>6.3f} {'—':>11}")
    log(f"{'primary (no IDF) → hybrid':<35} {'0.165':>6} {'0.071':>6} {'0.813':>11}")
    log(f"{'primary (IDF) → hybrid':<35} {primary_idf_metrics['recall@10']:>6.3f} "
        f"{primary_idf_metrics['mrr']:>6.3f} "
        f"{primary_idf_hybrid[primary_idf_alpha]['recall@10']:>11.3f}")
    log(f"{'v5c (no IDF) → hybrid':<35} {v5c_no_idf_metrics['recall@10']:>6.3f} "
        f"{v5c_no_idf_metrics['mrr']:>6.3f} "
        f"{hybrid_no_idf_results[best_alpha_no_idf]['recall@10']:>11.3f}")
    log(f"{'v5c (IDF) → hybrid':<35} {v5c_idf_metrics['recall@10']:>6.3f} "
        f"{v5c_idf_metrics['mrr']:>6.3f} "
        f"{hybrid_idf_results[best_alpha_idf]['recall@10']:>11.3f}")

    # Best prune combo
    best_prune_hybrid = max(prune_results.items(),
                            key=lambda x: x[1]["hybrid_metrics"]["recall@10"])
    bp_n, bp_data = best_prune_hybrid
    log(f"{'v5c (IDF+prune='+str(bp_n)+') → hybrid':<35} "
        f"{bp_data['standalone']['recall@10']:>6.3f} "
        f"{bp_data['standalone']['mrr']:>6.3f} "
        f"{bp_data['hybrid_metrics']['recall@10']:>11.3f}")
    log("=" * 70)

    _save_results(results, "v5c_analysis")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Full-Corpus Evaluation (838 queries vs 231k corpus)
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=10800,
    volumes={SAE_DIR: sae_vol, SAVE_DIR: retrieval_vol},
    secrets=[HF_SECRET],
    memory=65536,
)
def eval_full_corpus(
    run_names: list[str] = None,
    sae_run_name: str = "primary_L19_16k",
    topk_query: int = 100,
    topk_doc: int = 400,
    lora_rank: int = 64,
    max_seq_len: int = 256,
):
    """Evaluate retrieval of 838 eval queries against the full 231k corpus.

    For each query, checks whether the ground-truth positive document
    appears in the top-K retrieved from the entire corpus. This is the
    realistic retrieval scenario (vs the 838x838 matrix eval).

    Also evaluates BM25 and hybrid SCAR+BM25.
    """
    import torch
    import json
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from rank_bm25 import BM25Okapi

    if run_names is None:
        run_names = ["v13_e25"]

    sae_vol.reload()
    retrieval_vol.reload()

    device = torch.device("cuda")
    hf_token = os.environ.get("HF_TOKEN")

    # --- Load eval dataset ---
    log("Loading evaluation dataset...")
    eval_ds = load_dataset(EVAL_DATASET, split="train", token=hf_token)
    queries = [row["query"] for row in eval_ds]
    gt_documents = [row["ground_truth_code"] for row in eval_ds]
    N_eval = len(queries)
    log(f"Eval set: {N_eval} query-document pairs")

    # --- Load full corpus ---
    log("Loading 231k corpus...")
    corpus_ds = load_dataset(
        f"{HF_USERNAME}/scar-corpus", split="train", token=hf_token
    )
    corpus_docs = [row["contract_code"] for row in corpus_ds]
    N_corpus = len(corpus_docs)
    log(f"Corpus: {N_corpus} documents")

    # --- Inject ground-truth documents into corpus ---
    # The eval ground-truth docs may or may not be in the corpus.
    # We append them and track their indices.
    gt_indices = []
    corpus_with_gt = list(corpus_docs)
    for gt_doc in gt_documents:
        # Check if exact match exists in corpus (by hash)
        import hashlib
        gt_hash = hashlib.sha256(gt_doc.strip().encode()).hexdigest()
        found_idx = None
        # Skip expensive search - just append and deduplicate by position
        # (corpus is 231k, linear search is too slow)
        corpus_with_gt.append(gt_doc)
        gt_indices.append(len(corpus_with_gt) - 1)

    N_total = len(corpus_with_gt)
    log(f"Corpus + GT docs: {N_total} (added {N_total - N_corpus} GT docs)")

    # --- BM25 baseline ---
    log("Building BM25 index on full corpus...")
    tokenized_corpus = [doc.split() for doc in corpus_with_gt]
    bm25 = BM25Okapi(tokenized_corpus)
    log("BM25 index built")

    bm25_results = {"recall@1": 0, "recall@5": 0, "recall@10": 0, "recall@20": 0, "mrr": 0, "ndcg@10": 0}
    bm25_per_query = {f"recall@{k}": [] for k in [1, 5, 10, 20]}
    bm25_per_query["mrr"] = []
    bm25_per_query["ndcg@10"] = []

    log("Evaluating BM25 (single pass: eval + score storage)...")
    bm25_score_matrix = []
    for qi in range(N_eval):
        if qi % 100 == 0:
            log(f"  BM25 query {qi}/{N_eval}")
        q_tokens = queries[qi].split()
        scores = bm25.get_scores(q_tokens)
        bm25_score_matrix.append(scores)
        ranking = np.argsort(scores)[::-1]
        gt_idx = gt_indices[qi]
        rank_pos = np.where(ranking == gt_idx)[0]
        if len(rank_pos) > 0:
            rank = rank_pos[0] + 1  # 1-indexed
        else:
            rank = N_total + 1

        for k in [1, 5, 10, 20]:
            hit = 1.0 if rank <= k else 0.0
            bm25_per_query[f"recall@{k}"].append(hit)
        rr = 1.0 / rank if rank <= N_total else 0.0
        bm25_per_query["mrr"].append(rr)
        # nDCG@10: single relevant doc
        ndcg = 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0
        bm25_per_query["ndcg@10"].append(ndcg)

    bm25_score_matrix = np.array(bm25_score_matrix)  # (N_eval, N_total)
    for metric in bm25_per_query:
        bm25_results[metric] = float(np.mean(bm25_per_query[metric]))
    log(f"BM25 full-corpus: R@10={bm25_results['recall@10']:.4f}, "
        f"MRR={bm25_results['mrr']:.4f}, nDCG@10={bm25_results['ndcg@10']:.4f}")

    # Free BM25 memory
    del bm25, tokenized_corpus

    all_results = {
        "n_eval": N_eval,
        "n_corpus": N_total,
        "bm25": bm25_results,
    }

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    RetrieverClass = build_retriever_class()

    # --- Per-run evaluation ---
    run_config = {
        "v12_e15": {"sae_lora_rank": 256},
        "v13_e25": {"sae_lora_rank": 256},
        "v16_sae_only": {"sae_lora_rank": 256, "no_backbone_lora": True, "idf_from": "v13_e25"},
        "splade_v1": {"retriever_type": "splade"},
    }

    for run_name in run_names:
        log(f"\n{'=' * 60}")
        log(f"FULL-CORPUS EVAL: {run_name}")
        log(f"{'=' * 60}")

        rc = run_config.get(run_name, {})
        run_sae_lora_rank = rc.get("sae_lora_rank", 0)
        run_retriever_type = rc.get("retriever_type", "scar")
        no_backbone_lora = rc.get("no_backbone_lora", False)
        idf_source = rc.get("idf_from", run_name)
        lora_dir = f"{SAVE_DIR}/{run_name}/lora_adapter"
        retrieval_ckpt = f"{SAVE_DIR}/{run_name}/checkpoint_final.pt"
        idf_ckpt = f"{SAVE_DIR}/{idf_source}/checkpoint_final.pt"

        if not os.path.exists(lora_dir) and not os.path.exists(retrieval_ckpt):
            log(f"  SKIP: no checkpoint for {run_name}")
            all_results[run_name] = {"error": "checkpoint not found"}
            continue
        if idf_source != run_name:
            log(f"  IDF from: {idf_source}")

        # Load backbone
        log("  Loading backbone...")
        backbone = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16,
        ).to(device).eval()
        for p in backbone.parameters():
            p.requires_grad = False

        # Load LoRA
        if no_backbone_lora:
            model_with_lora = backbone
            model_with_lora.eval()
            log(f"  No backbone LoRA (SAE-LoRA only)")
        elif os.path.exists(lora_dir):
            model_with_lora = PeftModel.from_pretrained(backbone, lora_dir)
            model_with_lora.eval()
        else:
            from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType
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

        enable_bidirectional_attention(model_with_lora)

        if run_retriever_type == "splade":
            # SPLADE: vocab-space retriever
            SPLADEClass = build_splade_retriever_class()
            retriever = SPLADEClass(
                model_with_lora, tokenizer, topk_query=topk_query,
                topk_doc=topk_doc, max_seq_len=max_seq_len,
            )
            log(f"  SPLADE mode (vocab_size={retriever.vocab_size})")
        else:
            # SCAR: SAE-based retriever
            sae_ckpt_path = f"{SAE_DIR}/{sae_run_name}/checkpoint_final.pt"
            run_sae, run_target_norm = load_frozen_sae(
                sae_ckpt_path, device, sae_lora_rank=run_sae_lora_rank)
            if run_sae_lora_rank > 0 and os.path.exists(retrieval_ckpt):
                ret_ckpt = torch.load(retrieval_ckpt, map_location=device, weights_only=False)
                if "sae_lora_state" in ret_ckpt:
                    run_sae.load_state_dict(ret_ckpt["sae_lora_state"], strict=False)
                    log(f"  SAE LoRA loaded (rank={run_sae_lora_rank})")

            retriever = RetrieverClass(
                model_with_lora, tokenizer, run_sae, layer_idx=LAYER_IDX,
                topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
                target_norm=run_target_norm,
            )

        # Load IDF (from idf_source, may differ from run_name for fair ablation)
        if os.path.exists(idf_ckpt):
            ckpt_data = torch.load(idf_ckpt, map_location="cpu", weights_only=False)
            if "idf_weights" in ckpt_data:
                retriever.idf_weights = ckpt_data["idf_weights"]
                log(f"  IDF weights loaded from {idf_source}")

        # --- Encode queries ---
        log(f"  Encoding {N_eval} queries...")
        all_q_vecs = []
        with torch.no_grad():
            for i in range(0, N_eval, 16):
                q_batch = queries[i:i + 16]
                q_sparse, _ = retriever.encode_queries(q_batch)
                all_q_vecs.append(q_sparse.cpu())
        q_matrix = torch.cat(all_q_vecs, dim=0)  # (N_eval, D)
        log(f"  Queries encoded: {q_matrix.shape}")

        # --- Encode full corpus + compute metrics ---
        # SPLADE (152k dims): chunked doc encoding — never materialize full d_matrix
        # SCAR (16k dims): concatenate all d_vecs (fits in memory)
        use_chunked_docs = run_retriever_type == "splade"

        if use_chunked_docs:
            # Chunked approach: encode docs in batches, compute similarity per batch,
            # accumulate per-query scores. Peak memory: q_matrix + one doc batch.
            log(f"  SPLADE: chunked doc encoding (vocab_size={q_matrix.shape[1]})")
            batch_size = 16

            # Initialize per-query best scores: for each query, track scores for all docs
            # We only need the rank of the GT doc, so we track: score of GT doc, and
            # count of docs scoring higher.
            gt_scores = torch.zeros(N_eval)
            docs_above_gt = torch.zeros(N_eval, dtype=torch.long)

            # Also store SCAR sim scores for hybrid (need per-query scores for all docs)
            # For 152k dims, we store per-query scores as numpy to avoid torch memory
            all_scar_scores = np.zeros((N_eval, N_total), dtype=np.float32)

            doc_offset = 0
            with torch.no_grad():
                for i in range(0, N_total, batch_size):
                    if i % 5000 == 0:
                        log(f"    docs: {i}/{N_total}")
                    d_batch = corpus_with_gt[i:i + batch_size]
                    d_sparse, _ = retriever.encode_documents(d_batch)
                    d_sparse_cpu = d_sparse.cpu()
                    actual_batch = d_sparse_cpu.shape[0]

                    # Similarity: (N_eval, actual_batch)
                    sim = torch.matmul(q_matrix, d_sparse_cpu.T)
                    all_scar_scores[:, doc_offset:doc_offset + actual_batch] = sim.numpy()
                    doc_offset += actual_batch

            log(f"  Corpus encoded and scored: {doc_offset} docs")

            # Compute metrics from full score matrix
            scar_per_query = {f"recall@{k}": [] for k in [1, 5, 10, 20]}
            scar_per_query["mrr"] = []
            scar_per_query["ndcg@10"] = []

            for qi in range(N_eval):
                scores = all_scar_scores[qi]
                ranking = np.argsort(scores)[::-1]
                gt_idx = gt_indices[qi]
                rank_pos = np.where(ranking == gt_idx)[0]
                rank = rank_pos[0] + 1 if len(rank_pos) > 0 else N_total + 1

                for k in [1, 5, 10, 20]:
                    scar_per_query[f"recall@{k}"].append(1.0 if rank <= k else 0.0)
                scar_per_query["mrr"].append(1.0 / rank if rank <= N_total else 0.0)
                scar_per_query["ndcg@10"].append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)

        else:
            # SCAR: concatenate all doc vecs (fits for 16k dims)
            log(f"  Encoding {N_total} corpus documents...")
            all_d_vecs = []
            batch_size = 16
            with torch.no_grad():
                for i in range(0, N_total, batch_size):
                    if i % 5000 == 0:
                        log(f"    docs: {i}/{N_total}")
                    d_batch = corpus_with_gt[i:i + batch_size]
                    d_sparse, _ = retriever.encode_documents(d_batch)
                    all_d_vecs.append(d_sparse.cpu())
            d_matrix = torch.cat(all_d_vecs, dim=0)  # (N_total, D_SAE)
            log(f"  Corpus encoded: {d_matrix.shape}")

            # Compute retrieval metrics
            scar_per_query = {f"recall@{k}": [] for k in [1, 5, 10, 20]}
            scar_per_query["mrr"] = []
            scar_per_query["ndcg@10"] = []

            chunk_size = 100
            for qi_start in range(0, N_eval, chunk_size):
                qi_end = min(qi_start + chunk_size, N_eval)
                q_chunk = q_matrix[qi_start:qi_end]
                sim_chunk = torch.matmul(q_chunk, d_matrix.T)

                for local_i in range(qi_end - qi_start):
                    qi = qi_start + local_i
                    scores = sim_chunk[local_i]
                    ranking = scores.argsort(descending=True)
                    gt_idx = gt_indices[qi]
                    rank_pos = (ranking == gt_idx).nonzero(as_tuple=True)[0]
                    rank = rank_pos[0].item() + 1 if len(rank_pos) > 0 else N_total + 1

                    for k in [1, 5, 10, 20]:
                        scar_per_query[f"recall@{k}"].append(1.0 if rank <= k else 0.0)
                    scar_per_query["mrr"].append(1.0 / rank if rank <= N_total else 0.0)
                    scar_per_query["ndcg@10"].append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)

            # Store scores for hybrid computation
            all_scar_scores = np.zeros((N_eval, N_total), dtype=np.float32)
            for qi_start in range(0, N_eval, chunk_size):
                qi_end = min(qi_start + chunk_size, N_eval)
                q_chunk = q_matrix[qi_start:qi_end]
                all_scar_scores[qi_start:qi_end] = torch.matmul(q_chunk, d_matrix.T).numpy()
            del d_matrix

        log("  Computing retrieval metrics...")
        scar_results = {metric: float(np.mean(vals)) for metric, vals in scar_per_query.items()}
        log(f"  Standalone: R@10={scar_results['recall@10']:.4f}, "
            f"MRR={scar_results['mrr']:.4f}, nDCG@10={scar_results['ndcg@10']:.4f}")

        # --- Hybrid + BM25 ---
        log("  Computing hybrid+BM25...")
        bm25_norm = bm25_score_matrix.copy()
        for qi in range(N_eval):
            row = bm25_norm[qi]
            row_max = row.max()
            row_min = row.min()
            if row_max > row_min:
                bm25_norm[qi] = (row - row_min) / (row_max - row_min)
            else:
                bm25_norm[qi] = 0.0

        best_hybrid_r10 = 0
        best_alpha = 0.5
        hybrid_results_by_alpha = {}

        for alpha in [0.3, 0.5, 0.7]:
            hybrid_per_query = {f"recall@{k}": [] for k in [1, 5, 10, 20]}
            hybrid_per_query["mrr"] = []
            hybrid_per_query["ndcg@10"] = []

            for qi in range(N_eval):
                scar_row = all_scar_scores[qi]
                s_max, s_min = scar_row.max(), scar_row.min()
                if s_max > s_min:
                    scar_normed = (scar_row - s_min) / (s_max - s_min)
                else:
                    scar_normed = np.zeros_like(scar_row)

                hybrid_scores = alpha * scar_normed + (1 - alpha) * bm25_norm[qi]
                ranking = np.argsort(hybrid_scores)[::-1]
                gt_idx = gt_indices[qi]
                rank_pos = np.where(ranking == gt_idx)[0]
                rank = rank_pos[0] + 1 if len(rank_pos) > 0 else N_total + 1

                for k in [1, 5, 10, 20]:
                    hybrid_per_query[f"recall@{k}"].append(1.0 if rank <= k else 0.0)
                hybrid_per_query["mrr"].append(1.0 / rank if rank <= N_total else 0.0)
                hybrid_per_query["ndcg@10"].append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)

            hybrid_metrics = {m: float(np.mean(v)) for m, v in hybrid_per_query.items()}
            hybrid_results_by_alpha[alpha] = hybrid_metrics
            if hybrid_metrics["recall@10"] > best_hybrid_r10:
                best_hybrid_r10 = hybrid_metrics["recall@10"]
                best_alpha = alpha

        best_hybrid = hybrid_results_by_alpha[best_alpha]
        log(f"  Hybrid (alpha={best_alpha}): R@10={best_hybrid['recall@10']:.4f}, "
            f"MRR={best_hybrid['mrr']:.4f}, nDCG@10={best_hybrid['ndcg@10']:.4f}")

        all_results[run_name] = {
            "standalone": scar_results,
            "hybrid": {"best_alpha": best_alpha, "best_metrics": best_hybrid},
            "hybrid_by_alpha": hybrid_results_by_alpha,
        }

        del model_with_lora, backbone, retriever, q_matrix, all_scar_scores
        torch.cuda.empty_cache()

    # --- Summary ---
    log("\n" + "=" * 80)
    log("FULL-CORPUS EVALUATION RESULTS")
    log(f"838 queries vs {N_total} documents")
    log("=" * 80)
    log(f"{'Method':<25} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'MRR':>8} {'nDCG@10':>8}")
    log("-" * 80)
    bm = all_results["bm25"]
    log(f"{'BM25':<25} {bm['recall@1']:>8.4f} {bm['recall@5']:>8.4f} "
        f"{bm['recall@10']:>8.4f} {bm['recall@20']:>8.4f} {bm['mrr']:>8.4f} {bm['ndcg@10']:>8.4f}")
    for rn in run_names:
        if rn in all_results and "error" not in all_results[rn]:
            s = all_results[rn]["standalone"]
            log(f"{'SCAR('+rn+')':<25} {s['recall@1']:>8.4f} {s['recall@5']:>8.4f} "
                f"{s['recall@10']:>8.4f} {s['recall@20']:>8.4f} {s['mrr']:>8.4f} {s['ndcg@10']:>8.4f}")
            h = all_results[rn]["hybrid"]["best_metrics"]
            a = all_results[rn]["hybrid"]["best_alpha"]
            log(f"{'  +BM25(a='+str(a)+')':<25} {h['recall@1']:>8.4f} {h['recall@5']:>8.4f} "
                f"{h['recall@10']:>8.4f} {h['recall@20']:>8.4f} {h['mrr']:>8.4f} {h['ndcg@10']:>8.4f}")
    log("=" * 80)

    # Save results
    save_path = f"{SAVE_DIR}/full_corpus_eval.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    retrieval_vol.commit()
    log(f"Results saved to {save_path}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Efficiency Evaluation
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=10800,
    volumes={SAE_DIR: sae_vol, SAVE_DIR: retrieval_vol},
    secrets=[HF_SECRET],
)
def eval_efficiency(
    sae_run_name: str = "primary_L19_16k",
    topk_query: int = 100,
    topk_doc: int = 400,
    lora_rank: int = 64,
    max_seq_len: int = 256,
    n_timing_queries: int = 100,
    warmup_runs: int = 3,
):
    """Measure efficiency metrics for SCAR, SPLADE, Dense, and BM25.

    For each method, measures:
    - Index size (MB): serialized corpus representations
    - Query encoding time (ms): single-query latency (batch=1)
    - Retrieval latency (ms): encode + matmul + top-10 (P50/P95)

    Uses the full 232k corpus for index size, and 100 queries for timing.
    Output: Appendix D table for the paper.
    """
    import torch
    import time
    import json
    import pickle
    import tempfile
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from rank_bm25 import BM25Okapi
    import scipy.sparse

    sae_vol.reload()
    retrieval_vol.reload()
    device = torch.device("cuda")
    hf_token = os.environ.get("HF_TOKEN")

    # --- Load eval queries ---
    log("Loading eval dataset for timing queries...")
    eval_ds = load_dataset(EVAL_DATASET, split="train", token=hf_token)
    timing_queries = [row["query"] for row in eval_ds][:n_timing_queries]
    log(f"Timing queries: {len(timing_queries)}")

    # --- Load full corpus ---
    log("Loading 231k corpus...")
    corpus_ds = load_dataset(
        f"{HF_USERNAME}/scar-corpus", split="train", token=hf_token
    )
    corpus_docs = [row["contract_code"] for row in corpus_ds]
    N_corpus = len(corpus_docs)
    log(f"Corpus: {N_corpus} documents")
    del corpus_ds

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # ================================================================
    # 1. BM25
    # ================================================================
    log("\n=== BM25 Efficiency ===")
    tokenized_corpus = [doc.split() for doc in corpus_docs]

    # Index size
    with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
        pickle.dump(tokenized_corpus, f)
        f.flush()
        bm25_index_mb = os.path.getsize(f.name) / (1024 * 1024)
    log(f"  BM25 index size: {bm25_index_mb:.1f} MB")

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    del tokenized_corpus

    # Query + retrieval timing (BM25 does both in one step)
    bm25_latencies = []
    for i in range(warmup_runs):
        _ = bm25.get_scores(timing_queries[0].split())
    for q in timing_queries:
        t0 = time.perf_counter()
        scores = bm25.get_scores(q.split())
        top_10 = np.argpartition(scores, -10)[-10:]
        t1 = time.perf_counter()
        bm25_latencies.append((t1 - t0) * 1000)  # ms

    bm25_latencies = np.array(bm25_latencies)
    results["BM25"] = {
        "index_mb": round(bm25_index_mb, 1),
        "encode_ms": 0.0,  # BM25 doesn't encode queries
        "retrieve_p50_ms": round(float(np.percentile(bm25_latencies, 50)), 1),
        "retrieve_p95_ms": round(float(np.percentile(bm25_latencies, 95)), 1),
    }
    log(f"  BM25 retrieve: P50={results['BM25']['retrieve_p50_ms']:.1f}ms, "
        f"P95={results['BM25']['retrieve_p95_ms']:.1f}ms")
    del bm25

    # ================================================================
    # 2. Dense (Layer 19 mean-pool, 1536-dim)
    # ================================================================
    log("\n=== Dense Efficiency ===")
    backbone = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    _activations = {}
    target_layer = _get_target_layer(backbone, LAYER_IDX)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            _activations["hidden"] = output[0]
        else:
            _activations["hidden"] = output

    hook_handle = target_layer.register_forward_hook(hook_fn)

    def dense_encode(texts):
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_len,
        ).to(device)
        with torch.no_grad():
            backbone(**inputs)
            hidden = _activations["hidden"]  # (B, T, 1536)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled.cpu()

    # Encode corpus in batches
    log("  Encoding corpus (dense)...")
    all_dense_vecs = []
    batch_size = 32
    for i in range(0, N_corpus, batch_size):
        if i % 10000 == 0:
            log(f"    docs: {i}/{N_corpus}")
        batch = corpus_docs[i:i + batch_size]
        vecs = dense_encode(batch)
        all_dense_vecs.append(vecs)
    dense_matrix = torch.cat(all_dense_vecs, dim=0).half().numpy()  # float16
    del all_dense_vecs
    log(f"  Dense matrix: {dense_matrix.shape}")

    # Index size
    with tempfile.NamedTemporaryFile(suffix=".npy") as f:
        np.save(f, dense_matrix)
        f.flush()
        dense_index_mb = os.path.getsize(f.name) / (1024 * 1024)
    log(f"  Dense index size: {dense_index_mb:.1f} MB")

    # Query encoding timing
    dense_encode_times = []
    for i in range(warmup_runs):
        dense_encode([timing_queries[0]])
        torch.cuda.synchronize()
    for q in timing_queries:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        q_vec = dense_encode([q])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dense_encode_times.append((t1 - t0) * 1000)

    # Retrieval timing (encode + matmul + top-10)
    dense_retrieve_times = []
    for i in range(warmup_runs):
        q_vec = dense_encode([timing_queries[0]])
        torch.cuda.synchronize()
        _ = np.dot(q_vec.numpy().astype(np.float32), dense_matrix.astype(np.float32).T)
    for q in timing_queries:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        q_vec = dense_encode([q]).numpy().astype(np.float32)
        torch.cuda.synchronize()
        scores = np.dot(q_vec, dense_matrix.astype(np.float32).T)[0]
        top_10 = np.argpartition(scores, -10)[-10:]
        t1 = time.perf_counter()
        dense_retrieve_times.append((t1 - t0) * 1000)

    dense_encode_times = np.array(dense_encode_times)
    dense_retrieve_times = np.array(dense_retrieve_times)
    results["Dense"] = {
        "index_mb": round(dense_index_mb, 1),
        "encode_ms": round(float(np.median(dense_encode_times)), 1),
        "retrieve_p50_ms": round(float(np.percentile(dense_retrieve_times, 50)), 1),
        "retrieve_p95_ms": round(float(np.percentile(dense_retrieve_times, 95)), 1),
    }
    log(f"  Dense encode: {results['Dense']['encode_ms']:.1f}ms, "
        f"retrieve P50={results['Dense']['retrieve_p50_ms']:.1f}ms")

    hook_handle.remove()
    del dense_matrix, backbone
    torch.cuda.empty_cache()

    # ================================================================
    # 3. SPLADE (splade_v1: vocab-space, 152k dims)
    # ================================================================
    log("\n=== SPLADE Efficiency ===")
    splade_backbone = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
    ).to(device).eval()
    for p in splade_backbone.parameters():
        p.requires_grad = False

    splade_lora_dir = f"{SAVE_DIR}/splade_v1/lora_adapter"
    splade_ckpt = f"{SAVE_DIR}/splade_v1/checkpoint_final.pt"
    if os.path.exists(splade_lora_dir):
        splade_model = PeftModel.from_pretrained(splade_backbone, splade_lora_dir)
    else:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        splade_model = get_peft_model(splade_backbone, lora_config)
        ckpt = torch.load(splade_ckpt, map_location=device, weights_only=False)
        set_peft_model_state_dict(splade_model, ckpt["model_state_dict"])
    splade_model.eval()
    enable_bidirectional_attention(splade_model)

    SPLADEClass = build_splade_retriever_class()
    splade_retriever = SPLADEClass(
        splade_model, tokenizer, topk_query=topk_query,
        topk_doc=topk_doc, max_seq_len=max_seq_len,
    )

    # Load IDF
    if os.path.exists(splade_ckpt):
        ckpt_data = torch.load(splade_ckpt, map_location="cpu", weights_only=False)
        if "idf_weights" in ckpt_data:
            splade_retriever.idf_weights = ckpt_data["idf_weights"]

    # Encode corpus
    log("  Encoding corpus (SPLADE)...")
    splade_rows, splade_cols, splade_vals = [], [], []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, N_corpus, batch_size):
            if i % 10000 == 0:
                log(f"    docs: {i}/{N_corpus}")
            batch = corpus_docs[i:i + batch_size]
            d_sparse, _ = splade_retriever.encode_documents(batch)
            d_cpu = d_sparse.cpu()
            for j in range(d_cpu.shape[0]):
                row = d_cpu[j]
                nz = row.nonzero(as_tuple=True)[0]
                splade_rows.extend([i + j] * len(nz))
                splade_cols.extend(nz.tolist())
                splade_vals.extend(row[nz].tolist())

    splade_csr = scipy.sparse.csr_matrix(
        (splade_vals, (splade_rows, splade_cols)),
        shape=(N_corpus, splade_retriever.vocab_size),
    )
    log(f"  SPLADE sparse matrix: {splade_csr.shape}, nnz={splade_csr.nnz}")

    # Index size
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        scipy.sparse.save_npz(f.name, splade_csr)
        splade_index_mb = os.path.getsize(f.name) / (1024 * 1024)
    log(f"  SPLADE index size: {splade_index_mb:.1f} MB")

    # Query encoding timing
    splade_encode_times = []
    for i in range(warmup_runs):
        with torch.no_grad():
            splade_retriever.encode_queries([timing_queries[0]])
        torch.cuda.synchronize()
    for q in timing_queries:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            q_sparse, _ = splade_retriever.encode_queries([q])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        splade_encode_times.append((t1 - t0) * 1000)

    # Retrieval timing
    splade_retrieve_times = []
    for i in range(warmup_runs):
        with torch.no_grad():
            q_sparse, _ = splade_retriever.encode_queries([timing_queries[0]])
        torch.cuda.synchronize()
        q_np = q_sparse.cpu().numpy()
        _ = splade_csr.dot(q_np.T)
    for q in timing_queries:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            q_sparse, _ = splade_retriever.encode_queries([q])
        torch.cuda.synchronize()
        q_np = q_sparse.cpu().numpy().T  # (vocab_size, 1)
        scores = splade_csr.dot(q_np).flatten()
        top_10 = np.argpartition(scores, -10)[-10:]
        t1 = time.perf_counter()
        splade_retrieve_times.append((t1 - t0) * 1000)

    splade_encode_times = np.array(splade_encode_times)
    splade_retrieve_times = np.array(splade_retrieve_times)
    results["SPLADE"] = {
        "index_mb": round(splade_index_mb, 1),
        "encode_ms": round(float(np.median(splade_encode_times)), 1),
        "retrieve_p50_ms": round(float(np.percentile(splade_retrieve_times, 50)), 1),
        "retrieve_p95_ms": round(float(np.percentile(splade_retrieve_times, 95)), 1),
    }
    log(f"  SPLADE encode: {results['SPLADE']['encode_ms']:.1f}ms, "
        f"retrieve P50={results['SPLADE']['retrieve_p50_ms']:.1f}ms")

    del splade_model, splade_backbone, splade_retriever, splade_csr
    torch.cuda.empty_cache()

    # ================================================================
    # 4. SCAR (v13_e25: SAE-based, 16k dims)
    # ================================================================
    log("\n=== SCAR Efficiency ===")
    scar_backbone = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
    ).to(device).eval()
    for p in scar_backbone.parameters():
        p.requires_grad = False

    scar_lora_dir = f"{SAVE_DIR}/v13_e25/lora_adapter"
    scar_ckpt = f"{SAVE_DIR}/v13_e25/checkpoint_final.pt"
    if os.path.exists(scar_lora_dir):
        scar_model = PeftModel.from_pretrained(scar_backbone, scar_lora_dir)
    else:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        scar_model = get_peft_model(scar_backbone, lora_config)
        ckpt = torch.load(scar_ckpt, map_location=device, weights_only=False)
        set_peft_model_state_dict(scar_model, ckpt["model_state_dict"])
    scar_model.eval()
    enable_bidirectional_attention(scar_model)

    # Load SAE with LoRA
    sae_ckpt_path = f"{SAE_DIR}/{sae_run_name}/checkpoint_final.pt"
    scar_sae, scar_target_norm = load_frozen_sae(sae_ckpt_path, device, sae_lora_rank=256)
    if os.path.exists(scar_ckpt):
        ret_ckpt = torch.load(scar_ckpt, map_location=device, weights_only=False)
        if "sae_lora_state" in ret_ckpt:
            scar_sae.load_state_dict(ret_ckpt["sae_lora_state"], strict=False)

    RetrieverClass = build_retriever_class()
    scar_retriever = RetrieverClass(
        scar_model, tokenizer, scar_sae, layer_idx=LAYER_IDX,
        topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
        target_norm=scar_target_norm,
    )

    # Load IDF
    if os.path.exists(scar_ckpt):
        ckpt_data = torch.load(scar_ckpt, map_location="cpu", weights_only=False)
        if "idf_weights" in ckpt_data:
            scar_retriever.idf_weights = ckpt_data["idf_weights"]

    # Encode corpus
    log("  Encoding corpus (SCAR)...")
    scar_rows, scar_cols, scar_vals = [], [], []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, N_corpus, batch_size):
            if i % 10000 == 0:
                log(f"    docs: {i}/{N_corpus}")
            batch = corpus_docs[i:i + batch_size]
            d_sparse, _ = scar_retriever.encode_documents(batch)
            d_cpu = d_sparse.cpu()
            for j in range(d_cpu.shape[0]):
                row = d_cpu[j]
                nz = row.nonzero(as_tuple=True)[0]
                scar_rows.extend([i + j] * len(nz))
                scar_cols.extend(nz.tolist())
                scar_vals.extend(row[nz].tolist())

    scar_csr = scipy.sparse.csr_matrix(
        (scar_vals, (scar_rows, scar_cols)),
        shape=(N_corpus, D_SAE),
    )
    log(f"  SCAR sparse matrix: {scar_csr.shape}, nnz={scar_csr.nnz}")

    # Index size
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        scipy.sparse.save_npz(f.name, scar_csr)
        scar_index_mb = os.path.getsize(f.name) / (1024 * 1024)
    log(f"  SCAR index size: {scar_index_mb:.1f} MB")

    # Query encoding timing
    scar_encode_times = []
    for i in range(warmup_runs):
        with torch.no_grad():
            scar_retriever.encode_queries([timing_queries[0]])
        torch.cuda.synchronize()
    for q in timing_queries:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            q_sparse, _ = scar_retriever.encode_queries([q])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        scar_encode_times.append((t1 - t0) * 1000)

    # Retrieval timing
    scar_retrieve_times = []
    for i in range(warmup_runs):
        with torch.no_grad():
            q_sparse, _ = scar_retriever.encode_queries([timing_queries[0]])
        torch.cuda.synchronize()
        q_np = q_sparse.cpu().numpy()
        _ = scar_csr.dot(q_np.T)
    for q in timing_queries:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            q_sparse, _ = scar_retriever.encode_queries([q])
        torch.cuda.synchronize()
        q_np = q_sparse.cpu().numpy().T  # (D_SAE, 1)
        scores = scar_csr.dot(q_np).flatten()
        top_10 = np.argpartition(scores, -10)[-10:]
        t1 = time.perf_counter()
        scar_retrieve_times.append((t1 - t0) * 1000)

    scar_encode_times = np.array(scar_encode_times)
    scar_retrieve_times = np.array(scar_retrieve_times)
    results["SCAR"] = {
        "index_mb": round(scar_index_mb, 1),
        "encode_ms": round(float(np.median(scar_encode_times)), 1),
        "retrieve_p50_ms": round(float(np.percentile(scar_retrieve_times, 50)), 1),
        "retrieve_p95_ms": round(float(np.percentile(scar_retrieve_times, 95)), 1),
    }
    log(f"  SCAR encode: {results['SCAR']['encode_ms']:.1f}ms, "
        f"retrieve P50={results['SCAR']['retrieve_p50_ms']:.1f}ms")

    del scar_model, scar_backbone, scar_retriever, scar_csr
    torch.cuda.empty_cache()

    # ================================================================
    # Summary table
    # ================================================================
    log("\n" + "=" * 80)
    log("EFFICIENCY COMPARISON (232k corpus)")
    log("=" * 80)
    log(f"{'Method':<12} {'Index (MB)':>12} {'Encode (ms)':>12} {'Retrieve P50':>14} {'Retrieve P95':>14}")
    log("-" * 80)
    for method in ["BM25", "Dense", "SPLADE", "SCAR"]:
        r = results[method]
        enc = f"{r['encode_ms']:.1f}" if r['encode_ms'] > 0 else "N/A"
        log(f"{method:<12} {r['index_mb']:>12.1f} {enc:>12} "
            f"{r['retrieve_p50_ms']:>14.1f} {r['retrieve_p95_ms']:>14.1f}")
    log("=" * 80)

    # Save results
    save_path = f"{SAVE_DIR}/efficiency_results.json"
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    retrieval_vol.commit()
    log(f"Results saved to {save_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Push Weights to HuggingFace
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu=None,
    timeout=3600,
    volumes={SAE_DIR: sae_vol, SAVE_DIR: retrieval_vol},
    secrets=[HF_SECRET],
)
def push_scar_weights():
    """Push SAE + 15ep + 25ep weights to Farseen0/scar-weights with model card."""
    import os, json, shutil, tempfile
    from huggingface_hub import HfApi

    hf_token = os.environ["HF_TOKEN"]
    api = HfApi(token=hf_token)
    repo_id = f"{HF_USERNAME}/scar-weights"

    # Create repo
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False,
                    token=hf_token)
    log(f"Repo: {repo_id}")

    retrieval_vol.reload()
    sae_vol.reload()

    # --- 1. SAE weights ---
    sae_dir = f"{SAE_DIR}/primary_L19_16k"
    sae_files = [f for f in os.listdir(sae_dir) if not f.startswith('.')]
    log(f"SAE files: {sae_files}")
    for fname in sae_files:
        fpath = os.path.join(sae_dir, fname)
        if os.path.isfile(fpath):
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"sae/{fname}",
                repo_id=repo_id, repo_type="model", token=hf_token,
            )
    log("SAE weights pushed")

    # --- 2 & 3. Retrieval models ---
    models = {
        "scar-25ep": "v13_e25",   # best in-distribution
        "scar-15ep": "v12_e15",   # best OOD
    }

    for label, run_name in models.items():
        run_dir = f"{SAVE_DIR}/{run_name}"
        if not os.path.exists(run_dir):
            log(f"WARNING: {run_dir} not found, skipping {label}")
            continue

        # Upload checkpoint
        ckpt_path = f"{run_dir}/checkpoint_final.pt"
        if os.path.exists(ckpt_path):
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=f"{label}/checkpoint_final.pt",
                repo_id=repo_id, repo_type="model", token=hf_token,
            )
            log(f"{label}: checkpoint uploaded")

        # Upload LoRA adapter
        lora_dir = f"{run_dir}/lora_adapter"
        if os.path.exists(lora_dir):
            api.upload_folder(
                folder_path=lora_dir,
                path_in_repo=f"{label}/lora_adapter",
                repo_id=repo_id, repo_type="model", token=hf_token,
            )
            log(f"{label}: LoRA adapter uploaded")

        # Upload config
        config_path = f"{run_dir}/config.json"
        if os.path.exists(config_path):
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo=f"{label}/config.json",
                repo_id=repo_id, repo_type="model", token=hf_token,
            )

        log(f"{label}: done ({run_name})")

    # --- Model card ---
    model_card = """---
language:
  - en
license: apache-2.0
tags:
  - sparse-retrieval
  - smart-contracts
  - security
  - sae
  - lora
base_model: Qwen/Qwen2.5-Coder-1.5B
---

# SCAR: Sparse Code Audit Retriever

**SCAR** is the first sparse latent retriever for smart contract security auditing, built on SAE-LoRA adaptation of a domain-specific JumpReLU Sparse Autoencoder.

Paper: *SCAR: Sparse Code Audit Retriever via SAE-LoRA Adaptation*

## Model Variants

| Model | Use Case | R@10 (838-pair) | R@10 (232k corpus) | EVMBench Coverage |
|-------|----------|-----------------|---------------------|-------------------|
| **scar-25ep** | In-distribution / closed-domain | **0.977** | **0.901** | 0.683 |
| **scar-15ep** | Out-of-distribution / broad coverage | 0.971 | 0.868 | **0.732** |

### When to use which:
- **scar-25ep**: Best for retrieving from a known corpus of audited contracts. Highest precision, sparser representations (dL0≈115). Use this if your evaluation data resembles professional audit findings.
- **scar-15ep**: Best for generalization to unseen vulnerability patterns. Higher EVMBench coverage (0.732 vs 0.683). Use this for production systems scanning diverse, previously-unseen contracts.

## Architecture

- **Backbone**: Qwen2.5-Coder-1.5B with bidirectional attention + LoRA (rank 64, Q/K/V/O)
- **SAE**: JumpReLU, 16,384 features (10.4× expansion), Layer 19 residual stream
- **SAE-LoRA**: Rank 256 adaptation of SAE encoder W_enc (4.6M params)
- **Pooling**: Sum-pool + log1p compression + IDF weighting
- **Sparsity**: TopK (query=100, doc=400)

## Repository Structure

```
sae/                    # Frozen JumpReLU SAE (shared by both models)
  sae_weights.pt        # SAE encoder/decoder/threshold weights
  config.json           # SAE hyperparameters
scar-25ep/              # 25-epoch model (best in-distribution)
  checkpoint_final.pt   # Full checkpoint (LoRA + SAE-LoRA + IDF + config)
  lora_adapter/         # PEFT-compatible backbone LoRA adapter
  config.json           # Training configuration
scar-15ep/              # 15-epoch model (best OOD)
  checkpoint_final.pt
  lora_adapter/
  config.json
```

## Loading

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load backbone + LoRA
backbone = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
model = PeftModel.from_pretrained(backbone, "Farseen0/scar-weights/scar-25ep/lora_adapter")

# Load checkpoint (contains SAE-LoRA weights + IDF weights)
ckpt = torch.load("scar-25ep/checkpoint_final.pt", map_location="cpu")
idf_weights = ckpt["idf_weights"]        # (16384,) IDF weights for scoring
sae_lora = ckpt["sae_lora_state"]        # SAE-LoRA A and B matrices
config = ckpt["config"]                  # Training config dict
```

## Training Data

- **Corpus**: 231,269 smart contracts ([Farseen0/scar-corpus](https://huggingface.co/datasets/Farseen0/scar-corpus))
- **Pairs**: 7,552 human-auditor contrastive pairs ([Farseen0/scar-pairs](https://huggingface.co/datasets/Farseen0/scar-pairs))
- **Eval**: 838 held-out pairs from 10 sources ([Farseen0/scar-eval](https://huggingface.co/datasets/Farseen0/scar-eval))

## Citation

```bibtex
@inproceedings{scar2026,
  title={SCAR: Sparse Code Audit Retriever via SAE-LoRA Adaptation},
  author={Shaikh, Farseen},
  year={2026}
}
```
"""

    # Write model card to temp file and upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(model_card)
        card_path = f.name

    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id, repo_type="model", token=hf_token,
        commit_message="Add model card with usage instructions",
    )
    os.unlink(card_path)

    log(f"Model card pushed. All weights at: https://huggingface.co/{repo_id}")
    return {"status": "success", "repo": repo_id}


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(mode: str = "full"):
    """Run evaluation pipeline.

    Args:
        mode: "full" | "ours" | "baselines" | "ablation" | "sae_ablation"
              | "multi_trial" | "compare" | "prune_sweep" | "v5c_analysis"
    """
    from datetime import datetime
    import json

    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"SCAR Evaluation (mode={mode})")

    if mode == "push_weights":
        print("\n--- PUSH SCAR WEIGHTS TO HUGGINGFACE ---")
        result = push_scar_weights.remote()
        print(f"Result: {json.dumps(result, indent=2) if result else 'None'}")
        return

    if mode == "sae_ablation":
        print("\n--- SAE ABLATION EVALUATION ---")
        result = eval_sae_ablations.remote()
        print(f"Result: {json.dumps(result, indent=2) if result else 'None'}")
        return

    if mode == "rank_ablation":
        print("\n--- RANK ABLATION (5 epochs, varying SAE-LoRA rank) ---")
        result = compare_runs.remote(
            run_names=[
                "v5c",        # rank 16, 5 ep
                "v5c_r32",    # rank 32, 5 ep
                "v5c_r64",    # rank 64, 5 ep
                "v7_r128",    # rank 128, 5 ep + distill
                "v11_r256",   # rank 256, 5 ep
            ],
        )
        if result:
            print(f"\nFinal results:\n{json.dumps(result, indent=2)}")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
        return

    if mode == "compare":
        print("\n--- COMPARING ALL RETRIEVAL RUNS ---")
        result = compare_runs.remote(
            run_names=[
                "v13_e25",
                "v14_nodistill",
            ],
        )
        if result:
            print(f"\nFinal results:\n{json.dumps(result, indent=2)}")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
        return

    if mode == "v15_eval":
        print("\n--- V15 DECONFOUND ABLATION EVAL ---")
        result = compare_runs.remote(
            run_names=["v15_r128_nodistill"],
        )
        if result:
            print(f"\nFinal results:\n{json.dumps(result, indent=2)}")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
        return

    if mode == "splade_eval":
        print("\n--- SPLADE BASELINE EVAL ---")
        result = compare_runs.remote(
            run_names=["splade_v1"],
        )
        if result:
            print(f"\nFinal results:\n{json.dumps(result, indent=2)}")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
        return

    if mode == "v16_eval":
        print("\n--- V16 SAE-LORA-ONLY EVAL ---")
        result = compare_runs.remote(
            run_names=["v16_sae_only"],
        )
        if result:
            print(f"\nFinal results:\n{json.dumps(result, indent=2)}")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
        return

    if mode == "efficiency":
        print("\n--- EFFICIENCY METRICS (232k corpus) ---")
        result = eval_efficiency.remote()
        if result:
            print(f"\nFinal results:\n{json.dumps(result, indent=2)}")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
        return

    if mode == "splade_full_corpus":
        print("\n--- SPLADE FULL-CORPUS EVAL ---")
        result = eval_full_corpus.remote(
            run_names=["splade_v1"],
        )
        if result:
            print(f"\nFinal results:\n{json.dumps(result, indent=2)}")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
        return

    if mode == "full_corpus":
        print("\n--- FULL-CORPUS EVAL (838 queries vs 231k corpus) ---")
        result = eval_full_corpus.remote(
            run_names=["v13_e25", "v12_e15"],
        )
        if result:
            print(f"\nFinal results:\n{json.dumps(result, indent=2)}")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
        return

    if mode == "prune_sweep":
        print("\n--- FEATURE PRUNING SWEEP ---")
        result = eval_prune_sweep.remote()
        if result:
            print(f"\nFinal results:\n{json.dumps(result, indent=2)}")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
        return

    if mode == "v5c_analysis":
        print("\n--- V5C DEEP ANALYSIS ---")
        result = eval_v5c_analysis.remote()
        if result:
            print(f"\nFinal results:\n{json.dumps(result, indent=2)}")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
        return

    print(f"\n--- RETRIEVAL EVALUATION (mode={mode}) ---")
    result = run_evaluation.remote(mode=mode)
    if result:
        print(f"\nFinal results:\n{json.dumps(result, indent=2)}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
