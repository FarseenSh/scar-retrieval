"""
SCAR Step 6: Retrieval Fine-Tuning — Modal Labs
==========================================================
Trains a contrastive retrieval model by attaching LoRA adapters to
Qwen2.5-Coder-1.5B's attention projections, enabling bidirectional attention,
and using the frozen SAE from Step 5 as the sparse representation layer.

Architecture (all locked from PRD):
  - Backbone: Qwen2.5-Coder-1.5B (frozen, fp16)
  - SAE: JumpReLU from Step 5 (frozen)
  - LoRA: rank 64, q_proj/k_proj/v_proj/o_proj only (NO MLP)
  - Attention: Bidirectional (replaces causal mask)
  - Pooling: max-pool over tokens → TopK (q=40, d=400)
  - Loss: InfoNCE contrastive + DF-FLOPS (lambda=0.0001)

Usage:
  modal run scripts/step6_retrieval.py                          # primary training
  modal run scripts/step6_retrieval.py --mode temperature       # temperature sweep
  modal run scripts/step6_retrieval.py --mode all               # sweep then primary
  modal run scripts/step6_retrieval.py --mode improve_v2        # BM25 mining + synthetic pairs
  modal run scripts/step6_retrieval.py --mode improve_v3        # GradCache + temp schedule + clean data
"""

import modal
import os
import math

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("scar-retrieval-training")

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
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

SAE_DIR = "/sae_training"
SAVE_DIR = "/retrieval_training"
HF_SECRET = modal.Secret.from_name("huggingface-token")
HF_USERNAME = "Farseen0"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
PAIRS_DATASET = f"{HF_USERNAME}/scar-pairs"
EVAL_DATASET = f"{HF_USERNAME}/scar-eval"
D_IN = 1536    # Qwen2.5-Coder-1.5B hidden_size
D_SAE = 16384  # SAE dictionary width
LAYER_IDX = 19


def log(msg: str):
    """Timestamped logging with forced flush for Modal containers."""
    import sys
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: SAE Encoder (Encode-Only)
# ═══════════════════════════════════════════════════════════════════════════

def build_sae_encoder():
    """Returns a lightweight SAE encoder class (encode-only, no decoder/loss).
    Called inside Modal functions to avoid top-level torch import."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class JumpReLUSAEEncoder(nn.Module):
        """Frozen JumpReLU SAE encoder for retrieval, with optional LoRA.

        Only implements encode() — no decoder, no training loss.
        Loaded from Step 5 checkpoint and frozen.

        CRITICAL: Even though params are frozen (requires_grad=False),
        the encode computation IS differentiable w.r.t. input x.
        Gradients flow: loss → SAE encode → backbone → LoRA adapters.

        v5c SAE LoRA: When lora_rank > 0, applies low-rank adaptation to
        W_enc: W_eff = W_enc + lora_A @ lora_B. This adapts which features
        fire for retrieval discrimination, with only 2*rank*max(d_in,d_sae)
        extra parameters. W_enc, b_enc, b_dec, log_threshold stay frozen.

        NOTE on normalization: Step 5 trained the SAE with ActivationNormalizer
        (E[||x||]=sqrt(d_in)≈39.19). The retriever normalizes activations to
        target_norm before encoding. However, JumpReLU thresholds are still
        miscalibrated (L0<1/token even with normalization), so the retriever
        uses per-token TopK on z_pre instead of JumpReLU-thresholded z.
        The SAE's W_enc serves as a learned feature dictionary; TopK provides
        controlled sparsity that JumpReLU cannot deliver for bidirectional attn.
        """

        def __init__(self, d_in: int = D_IN, d_sae: int = D_SAE,
                     lora_rank: int = 0):
            super().__init__()
            self.d_in = d_in
            self.d_sae = d_sae
            self.lora_rank = lora_rank

            self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
            self.b_enc = nn.Parameter(torch.zeros(d_sae))
            self.b_dec = nn.Parameter(torch.zeros(d_in))  # needed for centering
            self.log_threshold = nn.Parameter(torch.zeros(d_sae))

            # SAE LoRA: low-rank adaptation of W_enc for retrieval
            if lora_rank > 0:
                self.lora_A = nn.Parameter(torch.empty(d_in, lora_rank))
                self.lora_B = nn.Parameter(torch.zeros(lora_rank, d_sae))
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                # B=zeros → initial W_eff = W_enc (no change at init)

        @property
        def threshold(self):
            return self.log_threshold.exp()

        def encode(self, x):
            """Encode activations to sparse SAE features.

            Args:
                x: (batch, d_in) residual stream activations

            Returns:
                z: (batch, d_sae) post-JumpReLU activations (sparse)
                z_pre: (batch, d_sae) pre-JumpReLU activations
            """
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
    """Load SAE from Step 5 checkpoint and freeze base parameters.

    CRITICAL: Returns (sae, target_norm). The SAE was trained with
    ActivationNormalizer that scaled activations to target_norm ≈ 39.19.
    Step 6 MUST normalize activations to the same scale before encoding,
    otherwise JumpReLU thresholds are miscalibrated:
    - Without normalization: bidirectional norms ~85, L0 = 0.6/token → dead
    - With normalization: scaled to ~39.19, L0 ≈ 40-80/token → discriminative

    Args:
        sae_checkpoint_path: path to Step 5's checkpoint_final.pt
        device: torch device
        sae_lora_rank: if >0, add LoRA adapters to W_enc (trainable)

    Returns:
        (sae, target_norm): JumpReLUSAEEncoder and target activation norm
    """
    import torch
    import math
    SAEEncoder = build_sae_encoder()
    ckpt = torch.load(sae_checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["sae_state_dict"]

    # Infer dimensions from checkpoint
    d_in = state["W_enc"].shape[0]
    d_sae = state["W_enc"].shape[1]

    # Extract normalizer from Step 5 (REQUIRED for threshold calibration)
    target_norm = math.sqrt(d_in)  # default: sqrt(1536) ≈ 39.19
    if "normalizer_state_dict" in ckpt:
        norm_state = ckpt["normalizer_state_dict"]
        running_mean_norm = norm_state.get("running_mean_norm")
        target_norm = norm_state.get("target_norm", target_norm)
        log(f"SAE checkpoint normalizer: running_mean_norm={running_mean_norm:.4f}, "
            f"target_norm={target_norm:.4f} (Step 6 will normalize to target_norm)")

    sae = SAEEncoder(d_in=d_in, d_sae=d_sae, lora_rank=sae_lora_rank).to(device)
    sae.W_enc.data.copy_(state["W_enc"])
    sae.b_enc.data.copy_(state["b_enc"])
    sae.b_dec.data.copy_(state["b_dec"])
    sae.log_threshold.data.copy_(state["log_threshold"])

    sae.eval()
    # Freeze base SAE params; LoRA A/B stay trainable if present
    for name, p in sae.named_parameters():
        if name.startswith("lora_"):
            p.requires_grad = True  # trainable SAE LoRA
        else:
            p.requires_grad = False  # frozen base SAE

    if sae_lora_rank > 0:
        n_lora = sum(p.numel() for p in sae.parameters() if p.requires_grad)
        log(f"SAE LoRA enabled: rank={sae_lora_rank}, {n_lora:,} trainable params")

    return sae, target_norm


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Bidirectional Attention Modification
# ═══════════════════════════════════════════════════════════════════════════

def enable_bidirectional_attention(model):
    """Replace Qwen2's causal attention with full bidirectional attention.

    Qwen2 applies causal masking via _update_causal_mask() on the base model.
    We monkey-patch this to return an all-zeros 4D additive mask (attend everywhere).
    Also forces eager attention to avoid SDPA's is_causal=True shortcut.

    MUST be called AFTER get_peft_model() since PEFT wraps the model structure.
    """
    import torch
    import types

    # Navigate through PEFT wrapping to find the Qwen2Model
    # PEFT: model.base_model (PeftModel) → .model (Qwen2ForCausalLM) → .model (Qwen2Model)
    # No PEFT: model (Qwen2ForCausalLM) → .model (Qwen2Model)
    if hasattr(model, 'peft_config'):
        qwen_model = model.base_model.model.model  # PEFT path
    else:
        qwen_model = model.model  # Qwen2ForCausalLM → Qwen2Model

    # Monkey-patch _update_causal_mask to return bidirectional mask with padding
    def _bidirectional_update_causal_mask(self, attention_mask, input_tensor,
                                          cache_position, past_key_values,
                                          output_attentions=False):
        batch_size, seq_len = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device
        # Start with all-zeros (full bidirectional attention)
        mask_4d = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=dtype, device=device)
        # Apply padding mask if provided: pad tokens should not be attended to
        if attention_mask is not None and attention_mask.dim() == 2:
            # (B, S) → (B, 1, 1, S) — broadcast over query and head dims
            padding_mask = attention_mask[:, None, None, :].to(dtype)
            min_val = torch.finfo(dtype).min
            mask_4d = mask_4d.masked_fill(padding_mask == 0, min_val)
        return mask_4d

    qwen_model._update_causal_mask = types.MethodType(
        _bidirectional_update_causal_mask, qwen_model
    )

    # Disable is_causal on all attention layers
    for layer in qwen_model.layers:
        if hasattr(layer, 'self_attn'):
            layer.self_attn.is_causal = False

    log("Bidirectional attention enabled (causal mask → all-zeros)")


def _get_target_layer(model, layer_idx):
    """Get the target decoder layer, handling PEFT model wrapping.

    After get_peft_model(), the layer path changes from
    model.model.layers[i] to model.base_model.model.model.layers[i].
    """
    # Try PEFT path first: model.base_model.model.model.layers[i]
    try:
        return model.base_model.model.model.layers[layer_idx]
    except (AttributeError, TypeError):
        pass
    # Non-PEFT: model.model.layers[i]
    try:
        return model.model.layers[layer_idx]
    except (AttributeError, TypeError):
        pass
    raise ValueError(f"Cannot find layer {layer_idx} in model of type {type(model)}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Retrieval Model Wrapper
# ═══════════════════════════════════════════════════════════════════════════

def build_retriever_class():
    """Returns the ScarRetriever class. Called inside Modal functions."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ScarRetriever(nn.Module):
        """Wraps backbone + LoRA + frozen SAE + max-pool + TopK.

        Forward path:
        1. Tokenize → backbone (LoRA + bidirectional attention)
        2. Hook captures Layer 19 residual stream (NOT detached)
        3. Frozen SAE encodes → sparse 16k per token
        4. Max-pool across tokens → single 16k vector
        5. TopK selection → final sparse representation

        CRITICAL: Hook does NOT detach activations. Gradients must flow
        from loss through SAE encode through hook back to LoRA adapters.
        """

        def __init__(self, model, tokenizer, sae, layer_idx=LAYER_IDX,
                     topk_query=40, topk_doc=400, max_seq_len=256,
                     target_norm=None, per_token_k=64, pooling_mode="sum"):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self.sae = sae  # frozen
            self.layer_idx = layer_idx
            self.topk_query = topk_query
            self.topk_doc = topk_doc
            self.max_seq_len = max_seq_len
            self.target_norm = target_norm  # activation norm SAE was trained with
            self.per_token_k = per_token_k  # TopK per token (replaces JumpReLU)
            self.pooling_mode = pooling_mode  # "sum" or "max"
            self._activations = None
            self._logged_first_batch = False
            self.idf_weights = None  # (d_sae,) IDF weights, set via compute_idf()
            self.prune_mask = None   # (d_sae,) bool mask, set via set_prune_mask()

        def _encode_texts(self, texts, topk):
            """Encode a list of texts into sparse 16k vectors.

            Uses z_pre (SAE projections) with per-token TopK to handle the
            distribution shift from causal (Step 5) to bidirectional attention.

            Key insight: JumpReLU thresholds calibrated for causal attention are
            fundamentally miscalibrated for bidirectional activations — even with
            normalization, L0 drops from ~50/token to <1/token. We bypass JumpReLU
            entirely and use per-token TopK for controlled sparsity:
            - z_pre = (x - b_dec) @ W_enc + b_enc: projection onto learned features
            - ReLU: keep only positive projections (feature presence)
            - Per-token TopK(K): each token keeps K strongest features
            - Max-pool: aggregate per document
            - Doc-level TopK: final sparse representation

            Paper: "We use the SAE's encoder weights as a learned feature dictionary
            but replace JumpReLU with per-token TopK to bridge the distribution
            shift from causal pretraining to bidirectional retrieval fine-tuning."

            Gradient path: loss → L2-norm → doc-TopK scatter → log1p
            → sum-pool → token-TopK scatter → ReLU → z_pre → backbone → LoRA.

            Args:
                texts: list of strings
                topk: number of top features to keep per document

            Returns:
                sparse: (B, d_sae) with topk non-zero entries (for similarity)
                pooled: (B, d_sae) pre-doc-topk vectors (for DF-FLOPS)
            """
            import torch

            device = next(self.model.parameters()).device

            # Tokenize
            enc = self.tokenizer(
                texts, return_tensors="pt", truncation=True,
                max_length=self.max_seq_len, padding=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask_2d = enc["attention_mask"].to(device)

            # Register hook on target layer (NOT detaching — gradients flow through)
            target_layer = _get_target_layer(self.model, self.layer_idx)

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # float() for SAE numerical stability, NO detach for gradient flow
                self._activations = hidden.float()

            handle = target_layer.register_forward_hook(hook_fn)

            # Forward through INNER transformer model only (skip lm_head to save memory).
            # LoRA adapters live inside the layer modules, so they fire regardless.
            # Skipping lm_head saves batch_size * seq_len * vocab_size * 4 bytes
            # (e.g., 64 * 256 * 152k * 4 = 10GB for batch=64).
            try:
                if hasattr(self.model, 'peft_config'):
                    # PEFT path: PeftModel → LoraModel → Qwen2ForCausalLM → Qwen2Model
                    inner = self.model.base_model.model.model
                else:
                    inner = self.model.model
                inner(input_ids=input_ids, attention_mask=attention_mask_2d,
                      use_cache=False)
            finally:
                handle.remove()

            # activations: (batch, seq_len, d_in) — with gradient
            acts = self._activations  # (B, S, 1536)
            batch_size, seq_len, d_in = acts.shape

            # Normalize activations to the scale SAE was trained on.
            # Step 5 trained with ActivationNormalizer (target_norm ≈ 39.19).
            # Scale factor is detached to prevent normalization gradients.
            raw_acts = acts  # keep reference for diagnostics
            if self.target_norm is not None:
                with torch.no_grad():
                    batch_mean_norm = acts.norm(dim=-1).mean().clamp(min=1e-8)
                    norm_scale = self.target_norm / batch_mean_norm
                acts = acts * norm_scale  # gradient flows through acts, not scale

            # SAE encode: flatten → encode → get z_pre (pre-threshold projections)
            flat_acts = acts.reshape(-1, d_in)
            z, z_pre = self.sae.encode(flat_acts)  # (B*S, d_sae)

            # === Per-token TopK on z_pre (bypasses JumpReLU thresholds) ===
            # z_pre = projection onto each SAE feature direction
            # ReLU keeps only positive projections (feature presence)
            # TopK(K) per token gives controlled sparsity
            z_pre_3d = z_pre.reshape(batch_size, seq_len, -1)  # (B, S, d_sae)
            z_pre_pos = F.relu(z_pre_3d)  # (B, S, d_sae) — non-negative

            # Per-token TopK: keep only K strongest features per token
            # This replaces JumpReLU thresholds which are miscalibrated for
            # bidirectional attention (L0 < 1/token with JumpReLU vs K/token here)
            tok_topk_vals, tok_topk_idx = torch.topk(
                z_pre_pos, self.per_token_k, dim=-1,
            )  # (B, S, K)
            z_sparse = torch.zeros_like(z_pre_pos).scatter(
                -1, tok_topk_idx, tok_topk_vals,
            )  # (B, S, d_sae) with K non-zeros per token

            # Mask padding tokens (z_sparse >= 0, so zeros won't affect sums)
            pad_mask = attention_mask_2d.unsqueeze(-1).float()  # (B, S, 1)
            z_sparse = z_sparse * pad_mask

            # Pool over token positions
            if self.pooling_mode == "max":
                # Max-pool (SPLARE-style): keeps strongest activation per feature.
                # Preserves rare discriminative features that fire strongly on
                # 1-2 tokens (e.g. reentrancy pattern) instead of diluting them
                # with common features that fire on many tokens.
                pooled_raw = torch.amax(z_sparse, dim=1)  # (B, d_sae)
            else:
                # Sum-pool (SPLADE-style): captures both strength and frequency.
                pooled_raw = z_sparse.sum(dim=1)  # (B, d_sae)

            # SPLADE-style log saturation: s_j = log(1 + Σ_t ReLU(w_j^T h_t))
            # Compresses dynamic range so L2-norm doesn't collapse docs.
            # log1p also breaks proportionality: log(1+cx) ≠ c·log(1+x).
            pooled = torch.log1p(pooled_raw)

            # IDF weighting: down-weight features that appear in many documents
            # Without IDF, universal "this is Solidity code" features dominate TopK,
            # making all documents 97.5% similar (representation collapse).
            if self.idf_weights is not None:
                pooled = pooled * self.idf_weights.to(pooled.device)

            # Feature pruning: zero out most generic features (lowest IDF)
            if self.prune_mask is not None:
                pooled = pooled * self.prune_mask.to(pooled.device).float()

            # One-time diagnostic: verify encoding quality
            if not self._logged_first_batch:
                self._logged_first_batch = True
                with torch.no_grad():
                    raw_norm = raw_acts.norm(dim=-1).mean().item()
                    norm_after = acts.norm(dim=-1).mean().item()
                    # z stats (post-JumpReLU — NOT used, for reference)
                    z_nnz = (z > 0).float().sum(dim=-1).mean().item()
                    # z_pre stats
                    z_pre_pos_count = (z_pre_pos > 0).float().sum(dim=-1).mean().item()
                    # per-token sparse stats
                    tok_sparse_l0 = (z_sparse > 0).float().sum(dim=-1).mean().item()
                    # pooled stats (after max-pool + log1p)
                    pooled_l0 = (pooled > 0).float().sum(dim=-1).mean().item()
                    pooled_max = pooled.max().item()
                    pooled_mean_act = pooled[pooled > 0].mean().item() if (pooled > 0).any() else 0.0
                    raw_max = pooled_raw.max().item()
                    raw_mean = pooled_raw[pooled_raw > 0].mean().item() if (pooled_raw > 0).any() else 0.0
                    tgt = f", target={self.target_norm:.4f}" if self.target_norm else ""
                    log(f"[DIAG] First batch SAE encoding (per-token TopK + {self.pooling_mode}-pool + log1p):")
                    log(f"  activation norm: {raw_norm:.4f} → {norm_after:.4f}{tgt}")
                    log(f"  z_pre positive features/token: {z_pre_pos_count:.1f}")
                    log(f"  z (JumpReLU) L0/token:         {z_nnz:.1f} (NOT used)")
                    log(f"  per-token TopK({self.per_token_k}) L0/token: {tok_sparse_l0:.1f}")
                    log(f"  pooled L0 (post max-pool):     {pooled_l0:.1f} (per document)")
                    log(f"  pre-log1p: max={raw_max:.4f}, mean_active={raw_mean:.4f}")
                    log(f"  post-log1p: max={pooled_max:.4f}, mean_active={pooled_mean_act:.4f}")

            # Doc-level TopK selection (out-of-place scatter for clean autograd)
            topk_vals, topk_idx = torch.topk(pooled, topk, dim=-1)
            sparse = torch.zeros_like(pooled).scatter(-1, topk_idx, topk_vals)

            # L2-normalize sparse vectors for stable cosine similarity
            # Bounds dot products to [-1, 1], preventing loss divergence.
            # Gradient flows through normalization.
            sparse = F.normalize(sparse, dim=-1)

            # Cleanup
            self._activations = None

            return sparse, pooled

        def set_prune_mask(self, n_prune=0):
            """Zero out the n_prune most generic SAE features (lowest IDF).

            Like stop-word removal for sparse vectors. Requires IDF to be
            computed first. Set n_prune=0 to disable.
            """
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
            """Encode query texts with TopK=topk_query."""
            return self._encode_texts(texts, self.topk_query)

        def encode_documents(self, texts):
            """Encode document texts with TopK=400."""
            return self._encode_texts(texts, self.topk_doc)

        @torch.no_grad()
        def compute_idf(self, documents, batch_size=16):
            """Compute IDF weights from a corpus of documents.

            IDF_j = log(N / (1 + df_j)) where df_j is the number of documents
            where feature j has a non-zero pooled activation (pre-TopK).

            This must be called BEFORE training/eval. IDF weights are applied
            in _encode_texts before TopK selection, ensuring discriminative
            features are prioritized over universal "this is code" features.
            """
            import math

            # Temporarily disable IDF to get raw pooled values
            old_idf = self.idf_weights
            self.idf_weights = None

            N = len(documents)
            d_sae = self.sae.d_sae
            doc_freq = torch.zeros(d_sae)

            log(f"Computing IDF from {N} documents...")
            for i in range(0, N, batch_size):
                batch = documents[i:i+batch_size]
                _, pooled = self._encode_texts(batch, d_sae)  # topk=d_sae to get ALL features
                # Count documents where each feature is active
                active = (pooled > 0).float().cpu()
                doc_freq += active.sum(dim=0)

            # IDF = log(N / (1 + df))
            idf = torch.log(torch.tensor(N, dtype=torch.float32) / (1.0 + doc_freq))
            # Clamp negative IDF (features in ALL docs get IDF ≈ 0)
            idf = idf.clamp(min=0.0)

            # Stats
            nonzero_idf = idf[idf > 0]
            zero_idf = (idf == 0).sum().item()
            log(f"IDF computed: {zero_idf} features with IDF=0 (appear in ALL docs)")
            log(f"IDF stats: mean={nonzero_idf.mean():.2f}, max={idf.max():.2f}, "
                f"nonzero={len(nonzero_idf)}/{d_sae}")

            self.idf_weights = idf
            # Restore old IDF if we had one
            if old_idf is not None:
                self.idf_weights = idf  # use fresh IDF

            return idf

    return ScarRetriever


def build_splade_retriever_class():
    """Returns a SPLADE retriever class that uses the LM head (vocab projection)
    instead of SAE encoder. Same backbone + LoRA, different projection layer.
    This is the key ablation: SAE features (16k) vs vocab-space (152k)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SPLADERetriever(nn.Module):
        """SPLADE-style retriever using LM head for vocab-space sparse representations.

        Forward path:
        1. Tokenize → backbone (LoRA + bidirectional attention)
        2. Full forward through lm_head → logits (B, S, vocab_size)
        3. SPLADE transform: log(1 + ReLU(logits)) per token
        4. Max-pool across tokens
        5. Optional IDF weighting
        6. Doc-level TopK selection
        7. L2 normalize
        """

        def __init__(self, model, tokenizer, topk_query=100, topk_doc=400,
                     max_seq_len=256):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self.topk_query = topk_query
            self.topk_doc = topk_doc
            self.max_seq_len = max_seq_len
            self.idf_weights = None
            self._logged_first_batch = False

            # Get vocab size from model config
            if hasattr(model, 'config'):
                self.vocab_size = model.config.vocab_size
            else:
                self.vocab_size = model.base_model.model.config.vocab_size

        def _encode_texts(self, texts, topk):
            """Encode texts into sparse vocab-space vectors via SPLADE transform.

            Returns:
                sparse: (B, vocab_size) L2-normalized TopK vectors
                pooled: (B, vocab_size) pre-TopK vectors (for DF-FLOPS)
            """
            import torch

            device = next(self.model.parameters()).device

            enc = self.tokenizer(
                texts, return_tensors="pt", truncation=True,
                max_length=self.max_seq_len, padding=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask_2d = enc["attention_mask"].to(device)

            # Full forward through model including lm_head
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask_2d,
                use_cache=False,
            )
            logits = outputs.logits  # (B, S, vocab_size)

            # SPLADE transform: per-token log-saturation
            w = torch.log1p(F.relu(logits))  # (B, S, vocab_size)

            # Mask padding tokens
            w = w * attention_mask_2d.unsqueeze(-1).float()

            # Max-pool over tokens (standard SPLADE pooling)
            pooled = torch.amax(w, dim=1)  # (B, vocab_size)

            # IDF weighting
            if self.idf_weights is not None:
                pooled = pooled * self.idf_weights.to(pooled.device)

            # Diagnostic logging
            if not self._logged_first_batch:
                self._logged_first_batch = True
                with torch.no_grad():
                    pooled_l0 = (pooled > 0).float().sum(dim=-1).mean().item()
                    pooled_max = pooled.max().item()
                    pooled_mean = pooled[pooled > 0].mean().item() if (pooled > 0).any() else 0.0
                    log(f"[DIAG] SPLADE first batch (vocab_size={self.vocab_size}):")
                    log(f"  pooled L0: {pooled_l0:.1f} (per document)")
                    log(f"  pooled max={pooled_max:.4f}, mean_active={pooled_mean:.4f}")

            # Doc-level TopK selection
            topk_vals, topk_idx = torch.topk(pooled, topk, dim=-1)
            sparse = torch.zeros_like(pooled).scatter(-1, topk_idx, topk_vals)

            # L2 normalize
            sparse = F.normalize(sparse, dim=-1)

            return sparse, pooled

        def encode_queries(self, texts):
            return self._encode_texts(texts, self.topk_query)

        def encode_documents(self, texts):
            return self._encode_texts(texts, self.topk_doc)

        @torch.no_grad()
        def compute_idf(self, documents, batch_size=16):
            """Compute IDF weights from documents on vocab_size dimensions."""
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

            nonzero_idf = idf[idf > 0]
            zero_idf = (idf == 0).sum().item()
            log(f"SPLADE IDF: {zero_idf} features with IDF=0, "
                f"mean={nonzero_idf.mean():.2f}, max={idf.max():.2f}")

            self.idf_weights = idf
            return idf

    return SPLADERetriever


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Loss Functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_infonce_loss(query_vecs, pos_vecs, hard_neg_vecs, temperature):
    """InfoNCE contrastive loss with in-batch + hard negatives.

    For batch size B:
    - In-batch scores: Q @ P^T → (B, B), diagonal = positives
    - Hard neg scores: element-wise dot → (B, 1)
    - Total: each query has 1 positive + (B-1) in-batch negatives + 1 hard negative

    Args:
        query_vecs: (B, D) sparse query representations
        pos_vecs: (B, D) sparse positive document representations
        hard_neg_vecs: (B, D) sparse hard negative representations
        temperature: scalar temperature tau

    Returns:
        loss: scalar InfoNCE loss
    """
    import torch
    import torch.nn.functional as F

    # In-batch similarity: (B, B) — diagonal = positive pairs
    in_batch_scores = torch.matmul(query_vecs, pos_vecs.T) / temperature

    # Hard negative similarity: (B, 1) — dedicated hard negative per query
    hard_neg_scores = (query_vecs * hard_neg_vecs).sum(dim=-1, keepdim=True) / temperature

    # Concatenate: (B, B+1) — positive at index i for query i
    all_scores = torch.cat([in_batch_scores, hard_neg_scores], dim=1)

    # Labels: positive is at diagonal position i
    labels = torch.arange(query_vecs.shape[0], device=query_vecs.device)

    return F.cross_entropy(all_scores, labels)


def compute_df_flops(representations):
    """DF-FLOPS sparsity regularization.

    Penalizes features that are active across many documents (bad for
    inverted index efficiency). Operates on pre-TopK representations.

    DF-FLOPS = sum of squared per-feature means across the batch.

    Args:
        representations: (B, D) pre-TopK pooled representations

    Returns:
        df_flops: scalar regularization term
    """
    mean_per_feature = representations.mean(dim=0)  # (D,)
    return (mean_per_feature ** 2).sum()


def compute_margin_mse(query_sparse, pos_sparse, neg_sparse, ce_pos_scores, ce_neg_scores):
    """Margin-MSE distillation loss from cross-encoder.

    Aligns the bi-encoder's margin (sim(q,pos) - sim(q,neg)) with the
    cross-encoder's margin. This is the key technique from SPLADE v2.

    Args:
        query_sparse: (B, D) sparse query representations
        pos_sparse: (B, D) sparse positive representations
        neg_sparse: (B, D) sparse hard negative representations
        ce_pos_scores: (B,) cross-encoder scores for (query, positive)
        ce_neg_scores: (B,) cross-encoder scores for (query, hard_negative)
    """
    import torch
    import torch.nn.functional as F

    # Bi-encoder margins (dot product similarity)
    bi_pos = (query_sparse * pos_sparse).sum(dim=-1)  # (B,)
    bi_neg = (query_sparse * neg_sparse).sum(dim=-1)  # (B,)
    bi_margin = bi_pos - bi_neg

    # Cross-encoder margin (teacher signal)
    ce_margin = ce_pos_scores - ce_neg_scores

    return F.mse_loss(bi_margin, ce_margin)


def compute_total_loss(query_sparse, pos_sparse, neg_sparse,
                       query_pooled, pos_pooled, neg_pooled,
                       temperature, lambda_d=0.0001, lambda_q=0.0001,
                       ce_pos_scores=None, ce_neg_scores=None, lambda_distill=0.0):
    """Combined retrieval loss: InfoNCE + DF-FLOPS + optional margin-MSE distillation.

    Returns:
        total_loss: scalar
        metrics: dict with component losses and quality indicators
    """
    import torch

    infonce = compute_infonce_loss(query_sparse, pos_sparse, neg_sparse, temperature)
    df_flops_doc = compute_df_flops(pos_pooled)
    df_flops_query = compute_df_flops(query_pooled)

    total = infonce + lambda_d * df_flops_doc + lambda_q * df_flops_query

    # Margin-MSE distillation from cross-encoder
    margin_mse = None
    if ce_pos_scores is not None and lambda_distill > 0:
        margin_mse = compute_margin_mse(
            query_sparse, pos_sparse, neg_sparse,
            ce_pos_scores, ce_neg_scores)
        total = total + lambda_distill * margin_mse

    # Compute quality metrics (no gradient)
    with torch.no_grad():
        q_l0 = (query_sparse > 0).float().sum(dim=-1).mean().item()
        d_l0 = (pos_sparse > 0).float().sum(dim=-1).mean().item()
        # Batch accuracy: is the positive the top-scored doc for each query?
        in_batch = torch.matmul(query_sparse, pos_sparse.T)
        labels = torch.arange(query_sparse.shape[0], device=query_sparse.device)
        acc = (in_batch.argmax(dim=1) == labels).float().mean().item()

    metrics = {
        "infonce": infonce.item(),
        "df_flops_doc": df_flops_doc.item(),
        "df_flops_query": df_flops_query.item(),
        "total": total.item(),
        "query_l0": q_l0,
        "doc_l0": d_l0,
        "batch_acc": acc,
    }
    if margin_mse is not None:
        metrics["margin_mse"] = margin_mse.item()
    return total, metrics


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Dataset & DataLoader
# ═══════════════════════════════════════════════════════════════════════════

def load_contrastive_dataset(hf_token=None, max_pairs=None, dataset_override=""):
    """Load contrastive pairs from HuggingFace.

    Columns: query, positive, hard_negative, negative_type, source,
             severity, vuln_type, quality_tier
    """
    from datasets import load_dataset
    ds_name = dataset_override if dataset_override else PAIRS_DATASET
    ds = load_dataset(ds_name, split="train", token=hf_token)
    if max_pairs is not None:
        ds = ds.select(range(min(max_pairs, len(ds))))
    log(f"Loaded {len(ds)} contrastive pairs")
    return ds


def create_collate_fn(return_indices=False):
    """Returns a collate function that groups raw dataset rows into string lists.
    Tokenization is deferred to the retriever's _encode_texts method.
    """
    def collate_fn(batch):
        queries = [row["query"] for row in batch]
        positives = [row["positive"] for row in batch]
        hard_negatives = [row["hard_negative"] for row in batch]
        if return_indices:
            indices = [row["_idx"] for row in batch]
            return queries, positives, hard_negatives, indices
        return queries, positives, hard_negatives
    return collate_fn


def create_dataloader(dataset, batch_size, shuffle=True):
    """Create DataLoader for contrastive training.

    drop_last=True ensures uniform batch size for in-batch negatives.
    num_workers=0 for Modal container compatibility.
    """
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=create_collate_fn(), num_workers=0, drop_last=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6b: BM25 Hard Negative Mining & Synthetic Pair Generation
# ═══════════════════════════════════════════════════════════════════════════

def mine_bm25_negatives(pairs_dataset, corpus_docs, n_negatives=4,
                        min_rank=5, max_rank=50):
    """Mine hard negatives from SAE corpus using BM25 for each training query.

    For each query, finds top-50 BM25 results from the corpus, filters out
    the known positive, and picks n_negatives from ranks min_rank..max_rank.
    These are "hard but not too hard" negatives — relevant enough to confuse
    a lexical matcher but semantically different from the actual vulnerability.

    SPLADE v2's biggest gain (+1.6 MRR) came from BM25-mined negatives.

    Args:
        pairs_dataset: HF dataset with query, positive, hard_negative columns
        corpus_docs: list of str — SAE corpus documents
        n_negatives: number of negatives to mine per query (4 → 4x expansion)
        min_rank: lowest BM25 rank to sample from (skip top results = too easy)
        max_rank: highest BM25 rank to sample from (below this = too hard)

    Returns:
        list of dicts with same schema as pairs_dataset rows
    """
    from rank_bm25 import BM25Okapi
    from tqdm import tqdm

    log(f"BM25 mining: tokenizing {len(corpus_docs)} corpus docs...")
    tokenized_corpus = [doc.split() for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    log("BM25 index built")

    expanded_rows = []
    skipped = 0

    for row_idx, row in enumerate(tqdm(pairs_dataset, desc="BM25 mining")):
        query_tokens = row["query"].split()
        scores = bm25.get_scores(query_tokens)
        ranked_indices = scores.argsort()[::-1]

        # Positive-aware filtering (NV-Retriever style):
        # Skip candidates whose BM25 score >= 95% of the positive's score
        # — these are likely false negatives (actually relevant but unlabeled)
        pos_prefix = row["positive"][:200]
        pos_score = max(scores[i] for i, doc in enumerate(corpus_docs)
                        if doc[:200] == pos_prefix) if any(
                            doc[:200] == pos_prefix for doc in corpus_docs
                        ) else 0.0
        score_threshold = 0.95 * pos_score if pos_score > 0 else float('inf')

        candidates = []
        for idx in ranked_indices[min_rank:max_rank]:
            if corpus_docs[idx][:200] != pos_prefix and scores[idx] < score_threshold:
                candidates.append(corpus_docs[idx])
            if len(candidates) >= n_negatives:
                break

        if len(candidates) == 0:
            skipped += 1
            continue

        for neg in candidates:
            expanded_rows.append({
                "query": row["query"],
                "positive": row["positive"],
                "hard_negative": neg,
                "negative_type": "bm25_mined",
                "source": row.get("source", "unknown"),
                "severity": row.get("severity", ""),
                "vuln_type": row.get("vuln_type", ""),
                "quality_tier": row.get("quality_tier", ""),
            })

        if (row_idx + 1) % 500 == 0:
            log(f"  BM25 mining: {row_idx+1}/{len(pairs_dataset)} queries, "
                f"{len(expanded_rows)} pairs mined")

    log(f"BM25 mining complete: {len(expanded_rows)} new pairs from "
        f"{len(pairs_dataset)} queries ({skipped} skipped)")
    return expanded_rows


# Vulnerability pattern templates for synthetic pair generation
VULN_PATTERNS = {
    "reentrancy": {
        "patterns": [".call{value:", ".call.value(", "call{value:"],
        "description": "Contract performs external call before updating state, "
                       "allowing recursive callback to drain funds.",
        "severity": "HIGH",
    },
    "unchecked_return": {
        "patterns": [".send(", ".transfer(", ".call("],
        "description": "Contract does not check return value of external call, "
                       "silently ignoring failures.",
        "severity": "MEDIUM",
    },
    "access_control": {
        "patterns": ["onlyOwner", "require(msg.sender ==", "tx.origin"],
        "description": "Contract has missing or insufficient access control, "
                       "allowing unauthorized users to call privileged functions.",
        "severity": "HIGH",
    },
    "integer_overflow": {
        "patterns": ["SafeMath", "unchecked {", "overflow"],
        "description": "Contract uses unchecked arithmetic that may overflow or "
                       "underflow, leading to unexpected token balances.",
        "severity": "HIGH",
    },
    "delegatecall": {
        "patterns": ["delegatecall(", ".delegatecall("],
        "description": "Contract uses delegatecall to untrusted target, allowing "
                       "arbitrary storage manipulation.",
        "severity": "CRITICAL",
    },
    "flash_loan": {
        "patterns": ["flashLoan(", "flashloan(", "FlashLoan"],
        "description": "Contract is vulnerable to flash loan price manipulation "
                       "attack via oracle dependency.",
        "severity": "HIGH",
    },
    "selfdestruct": {
        "patterns": ["selfdestruct(", "suicide("],
        "description": "Contract contains selfdestruct that can be triggered by "
                       "attacker to destroy contract and send funds.",
        "severity": "CRITICAL",
    },
    "timestamp_dependence": {
        "patterns": ["block.timestamp", "block.number", "now"],
        "description": "Contract uses block.timestamp for critical logic, which "
                       "can be manipulated by miners within a small window.",
        "severity": "LOW",
    },
}


def generate_synthetic_pairs(corpus_docs, bm25_index=None, max_pairs=8000):
    """Generate synthetic contrastive pairs from SAE corpus using vulnerability patterns.

    Scans corpus for contracts containing known vulnerability patterns and creates
    synthetic query-positive pairs. Hard negatives are BM25-mined from remaining corpus.

    Args:
        corpus_docs: list of str — SAE corpus documents
        bm25_index: pre-built BM25Okapi index (if None, builds one)
        max_pairs: maximum total pairs to generate

    Returns:
        list of dicts with contrastive pair schema
    """
    from rank_bm25 import BM25Okapi
    import random

    log(f"Synthetic pair generation: scanning {len(corpus_docs)} docs for vulnerability patterns...")

    if bm25_index is None:
        tokenized_corpus = [doc.split() for doc in corpus_docs]
        bm25_index = BM25Okapi(tokenized_corpus)

    # Find contracts matching each vulnerability pattern
    vuln_matches = {}  # vuln_type → list of (doc_idx, doc_text)
    for doc_idx, doc in enumerate(corpus_docs):
        doc_lower = doc.lower()
        for vuln_type, info in VULN_PATTERNS.items():
            for pattern in info["patterns"]:
                if pattern.lower() in doc_lower:
                    if vuln_type not in vuln_matches:
                        vuln_matches[vuln_type] = []
                    vuln_matches[vuln_type].append((doc_idx, doc))
                    break  # one match per vuln type per doc

    for vtype, matches in vuln_matches.items():
        log(f"  {vtype}: {len(matches)} matching contracts")

    # Generate pairs
    synthetic_pairs = []
    pairs_per_vuln = max_pairs // max(len(vuln_matches), 1)

    for vuln_type, matches in vuln_matches.items():
        info = VULN_PATTERNS[vuln_type]
        # Shuffle and cap per-type
        random.shuffle(matches)
        selected = matches[:pairs_per_vuln]

        for doc_idx, doc_text in selected:
            query = (f"{info['severity']} severity: {vuln_type.replace('_', ' ').title()}. "
                     f"{info['description']}")

            # BM25 hard negative: find similar-but-different contract
            query_tokens = query.split()
            scores = bm25_index.get_scores(query_tokens)
            ranked = scores.argsort()[::-1]

            hard_neg = None
            for neg_idx in ranked[5:50]:
                if neg_idx != doc_idx and corpus_docs[neg_idx][:200] != doc_text[:200]:
                    hard_neg = corpus_docs[neg_idx]
                    break

            if hard_neg is None:
                continue

            synthetic_pairs.append({
                "query": query,
                "positive": doc_text,
                "hard_negative": hard_neg,
                "negative_type": "synthetic_bm25",
                "source": "synthetic",
                "severity": info["severity"],
                "vuln_type": vuln_type,
                "quality_tier": 0,
            })

        if len(synthetic_pairs) >= max_pairs:
            break

    synthetic_pairs = synthetic_pairs[:max_pairs]
    log(f"Synthetic pair generation complete: {len(synthetic_pairs)} pairs")
    return synthetic_pairs


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6b: Cross-Encoder Training & Score Precomputation
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={SAVE_DIR: retrieval_vol},
    secrets=[HF_SECRET],
    timeout=7200,
)
def train_cross_encoder(
    lr: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 3,
    max_seq_len: int = 512,
    force: bool = False,
):
    """Train a cross-encoder on contrastive pairs for distillation.

    Uses Qwen2.5-Coder-1.5B + linear classification head.
    Binary cross-entropy: (query, positive) → 1, (query, hard_neg) → 0.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset
    import random
    import os
    import json

    save_dir = f"{SAVE_DIR}/cross_encoder"
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(f"{save_dir}/model_final.pt") and not force:
        log(f"Cross-encoder already trained at {save_dir}. Use force=True to retrain.")
        return {"status": "skipped"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Cross-encoder training: device={device}, GPU={torch.cuda.get_device_name()}")

    # Load model with classification head (num_labels=1 for regression-style scoring)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.train()
    log(f"Cross-encoder model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Load data
    hf_token = os.environ.get("HF_TOKEN")
    ds = load_dataset(PAIRS_DATASET, token=hf_token, split="train")
    log(f"Loaded {len(ds)} contrastive pairs")

    # Build training examples: (query, positive) → 1.0, (query, hard_neg) → 0.0
    examples = []
    for row in ds:
        q = row["query"]
        pos = row["positive"]
        neg = row.get("hard_negative", "")
        examples.append((q, pos, 1.0))
        if neg and len(neg.strip()) > 20:
            examples.append((q, neg, 0.0))
    random.shuffle(examples)
    log(f"Training examples: {len(examples)} (pos+neg)")

    # Split 95/5 for validation
    val_size = max(100, len(examples) // 20)
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_examples) // batch_size) * epochs
    log(f"Total steps: {total_steps}, val_size: {val_size}")

    step = 0
    for epoch in range(epochs):
        random.shuffle(train_examples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_examples), batch_size):
            batch = train_examples[i:i + batch_size]
            texts_a = [ex[0] for ex in batch]
            texts_b = [ex[1] for ex in batch]
            labels = torch.tensor([ex[2] for ex in batch], device=device, dtype=torch.float16)

            enc = tokenizer(
                texts_a, texts_b, padding=True, truncation=True,
                max_length=max_seq_len, return_tensors="pt"
            ).to(device)

            outputs = model(**enc)
            logits = outputs.logits.squeeze(-1).float()  # fp32 for stable loss
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            epoch_loss += loss.item()
            n_batches += 1

            if step % 100 == 0:
                log(f"  step {step}/{total_steps} | loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in range(0, len(val_examples), batch_size):
                batch = val_examples[i:i + batch_size]
                texts_a = [ex[0] for ex in batch]
                texts_b = [ex[1] for ex in batch]
                labels = torch.tensor([ex[2] for ex in batch], device=device)

                enc = tokenizer(
                    texts_a, texts_b, padding=True, truncation=True,
                    max_length=max_seq_len, return_tensors="pt"
                ).to(device)

                outputs = model(**enc)
                preds = (outputs.logits.squeeze(-1).float() > 0).float()
                val_correct += (preds == labels.float()).sum().item()
                val_total += len(labels)
        val_acc = val_correct / max(val_total, 1)
        model.train()
        log(f"Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}, val_acc={val_acc:.3f}")

    # Save model
    model.eval()
    torch.save(model.state_dict(), f"{save_dir}/model_final.pt")
    tokenizer.save_pretrained(save_dir)
    config = {
        "model_name": MODEL_NAME,
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "train_examples": len(train_examples),
        "val_acc": val_acc,
    }
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    # Delete stale distillation scores so they get re-computed on new data
    stale_scores = f"{save_dir}/distill_scores.pt"
    if os.path.exists(stale_scores):
        os.remove(stale_scores)
        log(f"Removed stale distillation scores (will be re-computed)")
    retrieval_vol.commit()
    log(f"Cross-encoder saved to {save_dir} (val_acc={val_acc:.3f})")
    return {"val_acc": val_acc, "train_loss": avg_loss}


@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={SAVE_DIR: retrieval_vol},
    secrets=[HF_SECRET],
    timeout=7200,
)
def precompute_ce_scores(max_seq_len: int = 512, batch_size: int = 32):
    """Pre-compute cross-encoder scores for all training pairs.

    Saves (ce_pos_scores, ce_neg_scores) tensors for margin-MSE distillation.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset
    import os

    device = torch.device("cuda")
    ce_dir = f"{SAVE_DIR}/cross_encoder"
    retrieval_vol.reload()

    if not os.path.exists(f"{ce_dir}/model_final.pt"):
        log("ERROR: Cross-encoder not trained yet. Run train_cross_encoder first.")
        return {"status": "error"}

    scores_path = f"{ce_dir}/distill_scores.pt"
    if os.path.exists(scores_path):
        log(f"Scores already computed at {scores_path}")
        scores = torch.load(scores_path, map_location="cpu")
        log(f"  pos_scores: {scores['pos_scores'].shape}, neg_scores: {scores['neg_scores'].shape}")
        return {"status": "exists", "n_pairs": scores["pos_scores"].shape[0]}

    # Load cross-encoder
    tokenizer = AutoTokenizer.from_pretrained(ce_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    state = torch.load(f"{ce_dir}/model_final.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log("Cross-encoder loaded")

    # Load training pairs
    hf_token = os.environ.get("HF_TOKEN")
    ds = load_dataset(PAIRS_DATASET, token=hf_token, split="train")
    queries = list(ds["query"])
    positives = list(ds["positive"])
    hard_negatives = list(ds["hard_negative"])
    log(f"Computing scores for {len(queries)} pairs...")

    def score_pairs(texts_a, texts_b):
        all_scores = []
        for i in range(0, len(texts_a), batch_size):
            ta = texts_a[i:i + batch_size]
            tb = texts_b[i:i + batch_size]
            enc = tokenizer(
                ta, tb, padding=True, truncation=True,
                max_length=max_seq_len, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits.squeeze(-1)
            all_scores.append(logits.float().cpu())
            if (i // batch_size) % 50 == 0:
                log(f"  scored {i}/{len(texts_a)}")
        return torch.cat(all_scores)

    pos_scores = score_pairs(queries, positives)
    neg_scores = score_pairs(queries, hard_negatives)

    # Normalize to [0, 1] range using sigmoid (logits → probabilities)
    pos_scores = torch.sigmoid(pos_scores)
    neg_scores = torch.sigmoid(neg_scores)

    margin = pos_scores - neg_scores
    log(f"Score stats: pos={pos_scores.mean():.3f}±{pos_scores.std():.3f}, "
        f"neg={neg_scores.mean():.3f}±{neg_scores.std():.3f}, "
        f"margin={margin.mean():.3f}±{margin.std():.3f}")
    log(f"Margin > 0: {(margin > 0).float().mean():.1%}")

    torch.save({"pos_scores": pos_scores, "neg_scores": neg_scores}, scores_path)
    retrieval_vol.commit()
    log(f"Saved distillation scores to {scores_path}")
    return {"status": "ok", "n_pairs": len(queries),
            "margin_mean": margin.mean().item(), "margin_positive_pct": (margin > 0).float().mean().item()}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: Main Training Function
# ═══════════════════════════════════════════════════════════════════════════

def _save_retrieval_checkpoint(path, model, optimizer, config, metrics,
                               step, epoch, total_steps, idf_weights=None,
                               sae_lora_state=None, has_peft=True):
    """Save a retrieval training checkpoint (LoRA weights only, not full backbone)."""
    import torch
    if has_peft:
        from peft import get_peft_model_state_dict
        model_state = get_peft_model_state_dict(model)
    else:
        model_state = {}  # No backbone LoRA — SAE LoRA saved separately
    ckpt = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
        "step": step,
        "epoch": epoch,
        "total_steps": total_steps,
    }
    if idf_weights is not None:
        ckpt["idf_weights"] = idf_weights.cpu()
    if sae_lora_state is not None:
        ckpt["sae_lora_state"] = {k: v.cpu() for k, v in sae_lora_state.items()}
    torch.save(ckpt, path)


@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={
        SAE_DIR: sae_vol,
        SAVE_DIR: retrieval_vol,
    },
    secrets=[HF_SECRET],
    timeout=86400,
)
def train_retrieval(
    temperature: float = 0.1,
    run_name: str = "primary",
    lr: float = 5e-5,
    batch_size: int = 32,
    micro_batch_size: int = 32,
    epochs: int = 5,
    max_seq_len: int = 256,
    warmup_ratio: float = 0.01,
    lambda_d: float = 1e-6,
    lambda_q: float = 1e-6,
    topk_query: int = 100,
    topk_doc: int = 400,
    lora_rank: int = 64,
    grad_clip: float = 1.0,
    log_every: int = 10,
    seed: int = None,
    checkpoint_pcts: list = None,
    sae_run_name: str = "primary_L19_16k",
    idf_corpus_size: int = 0,
    force: bool = False,
    use_bm25_mining: bool = False,
    bm25_corpus_cap: int = 50000,
    bm25_n_negatives: int = 4,
    synthetic_max_pairs: int = 8000,
    use_gradcache: bool = False,
    gradcache_chunk_size: int = 16,
    temp_schedule: str = "fixed",  # "fixed" or "cosine_cooldown"
    temp_max: float = 1.0,
    temp_min: float = 0.1,
    curriculum_easy_epochs: int = 0,  # 0=disabled, N=use random negatives for first N epochs
    pooling_mode: str = "sum",  # "sum" (SPLADE-style) or "max" (SPLARE-style)
    sae_lora_rank: int = 0,  # v5c: LoRA rank on SAE W_enc (0=frozen SAE)
    sae_lora_lr: float = 0.0,  # separate LR for SAE LoRA (0=use main lr)
    lambda_distill: float = 0.0,  # v7: margin-MSE distillation weight (0=disabled)
    pairs_dataset: str = "",  # override PAIRS_DATASET (empty=use default)
    retriever_type: str = "scar",  # "scar" (SAE-based) or "splade" (LM head vocab-space)
):
    """Train contrastive retrieval model with LoRA + frozen SAE.

    Trains LoRA adapters on Qwen2.5-Coder-1.5B attention projections using
    InfoNCE contrastive loss + DF-FLOPS sparsity regularization.
    """
    import torch
    import torch.nn.functional as F
    import json
    import shutil
    import random
    import numpy as np

    # Seed for reproducibility (but allow cuDNN non-determinism for optimization)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        log(f"{run_name}: seed={seed}")
    else:
        log(f"{run_name}: no seed (non-deterministic)")

    if checkpoint_pcts is None:
        checkpoint_pcts = [25, 50, 75, 100]

    sae_vol.reload()
    retrieval_vol.reload()

    run_dir = f"{SAVE_DIR}/{run_name}"
    final_ckpt = f"{run_dir}/checkpoint_final.pt"

    # Force re-run: delete stale results from previous failed runs
    if force and os.path.exists(run_dir):
        shutil.rmtree(run_dir)
        log(f"{run_name}: force=True, deleted stale run directory")

    # Skip if already done
    if os.path.exists(final_ckpt):
        log(f"{run_name}: already trained, skipping")
        return

    # Temperature is passed directly by caller (main → train_retrieval.remote).
    # When mode="all", main() runs sweep first and passes the result.
    # No auto-loading from sweep results — that caused tau=0.02 override bugs.
    log(f"{run_name}: temperature={temperature}")

    grad_accum_steps = batch_size // micro_batch_size

    os.makedirs(run_dir, exist_ok=True)

    # Save config
    config = {
        "run_name": run_name, "temperature": temperature, "lr": lr,
        "batch_size": batch_size, "micro_batch_size": micro_batch_size,
        "epochs": epochs, "max_seq_len": max_seq_len,
        "warmup_ratio": warmup_ratio, "lambda_d": lambda_d, "lambda_q": lambda_q,
        "topk_query": topk_query, "topk_doc": topk_doc, "seed": seed,
        "lora_rank": lora_rank, "grad_clip": grad_clip,
        "sae_run_name": sae_run_name, "model_name": MODEL_NAME,
        "dataset_name": PAIRS_DATASET, "grad_accum_steps": grad_accum_steps,
        "idf_corpus_size": idf_corpus_size,
        "use_bm25_mining": use_bm25_mining,
        "bm25_corpus_cap": bm25_corpus_cap,
        "bm25_n_negatives": bm25_n_negatives,
        "synthetic_max_pairs": synthetic_max_pairs,
        "use_gradcache": use_gradcache,
        "gradcache_chunk_size": gradcache_chunk_size,
        "temp_schedule": temp_schedule,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "curriculum_easy_epochs": curriculum_easy_epochs,
        "sae_lora_rank": sae_lora_rank,
        "retriever_type": retriever_type,
    }
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{run_name}: device={device}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    # --- Load backbone (frozen, fp16, eager attention for bidirectional) ---
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"{run_name}: loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    log(f"{run_name}: model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    # --- Apply LoRA adapters ---
    has_peft = lora_rank > 0
    if has_peft:
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,  # alpha=128, scaling factor = alpha/r = 2
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        log(f"{run_name}: LoRA applied (rank={lora_rank})")
    else:
        log(f"{run_name}: no backbone LoRA (lora_rank=0), backbone fully frozen")

    # Bidirectional attention: each token sees full context.
    # JumpReLU thresholds are bypassed (per-token TopK instead), so the
    # causal/bidirectional mismatch no longer matters for feature activation.
    enable_bidirectional_attention(model)

    # --- Build retriever ---
    sae = None
    if retriever_type == "splade":
        # SPLADE mode: use LM head (vocab-space, ~152k dims) instead of SAE
        SPLADERetrieverClass = build_splade_retriever_class()
        retriever = SPLADERetrieverClass(
            model, tokenizer, topk_query=topk_query, topk_doc=topk_doc,
            max_seq_len=max_seq_len,
        )
        sae_lora_rank = 0  # force no SAE LoRA
        log(f"{run_name}: SPLADE mode (vocab-space, {retriever.vocab_size} dims)")
    else:
        # SCAR mode: use SAE encoder (16k dims)
        sae_ckpt_path = f"{SAE_DIR}/{sae_run_name}/checkpoint_final.pt"
        if not os.path.exists(sae_ckpt_path):
            log(f"{run_name}: ERROR — SAE checkpoint not found at {sae_ckpt_path}")
            log(f"{run_name}: Contents of {SAE_DIR}: {os.listdir(SAE_DIR) if os.path.exists(SAE_DIR) else 'dir not found'}")
            return

        sae, target_norm = load_frozen_sae(sae_ckpt_path, device, sae_lora_rank=sae_lora_rank)
        log(f"{run_name}: SAE loaded from {sae_ckpt_path} (d_sae={sae.d_sae}, sae_lora_rank={sae_lora_rank})")

        RetrieverClass = build_retriever_class()
        retriever = RetrieverClass(
            model, tokenizer, sae, layer_idx=LAYER_IDX,
            topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
            target_norm=target_norm, pooling_mode=pooling_mode,
        )
        log(f"{run_name}: pooling_mode={pooling_mode}")

    # --- Dataset and dataloader ---
    hf_token = os.environ.get("HF_TOKEN")
    dataset = load_contrastive_dataset(hf_token=hf_token, dataset_override=pairs_dataset)

    # --- BM25 hard negative mining + synthetic pair generation ---
    if use_bm25_mining:
        from datasets import Dataset as HFDataset

        SAE_CORPUS = f"{HF_USERNAME}/scar-corpus"
        log(f"{run_name}: loading SAE corpus for BM25 mining (cap={bm25_corpus_cap})...")
        from datasets import load_dataset as _load_ds_bm25
        sae_corpus_ds = _load_ds_bm25(SAE_CORPUS, split="train", token=hf_token)
        corpus_size = min(bm25_corpus_cap, len(sae_corpus_ds))
        sae_corpus_ds = sae_corpus_ds.select(range(corpus_size))
        corpus_docs = [row["contract_code"] for row in sae_corpus_ds]
        del sae_corpus_ds
        log(f"{run_name}: loaded {len(corpus_docs)} corpus docs for BM25 mining")

        # Mine BM25 hard negatives (4x expansion) — with caching
        import json as _json_bm25
        bm25_cache_path = f"{run_dir}/bm25_mined_pairs.json"
        if os.path.exists(bm25_cache_path):
            log(f"{run_name}: loading cached BM25-mined pairs from {bm25_cache_path}")
            with open(bm25_cache_path) as f:
                bm25_pairs = _json_bm25.load(f)
            log(f"{run_name}: loaded {len(bm25_pairs)} cached BM25-mined pairs")
        else:
            log(f"{run_name}: mining BM25 hard negatives (n_negatives={bm25_n_negatives})...")
            bm25_pairs = mine_bm25_negatives(
                dataset, corpus_docs,
                n_negatives=bm25_n_negatives, min_rank=5, max_rank=50,
            )
            log(f"{run_name}: BM25 mining yielded {len(bm25_pairs)} pairs")
            # Cache to volume so we don't lose 1.5+ hrs of mining on crash
            with open(bm25_cache_path, "w") as f:
                _json_bm25.dump(bm25_pairs, f)
            retrieval_vol.commit()
            log(f"{run_name}: cached BM25-mined pairs to {bm25_cache_path}")

        # Generate synthetic pairs
        log(f"{run_name}: generating synthetic pairs (max={synthetic_max_pairs})...")
        synthetic_pairs = generate_synthetic_pairs(
            corpus_docs, bm25_index=None,
            max_pairs=synthetic_max_pairs,
        )
        log(f"{run_name}: synthetic generation yielded {len(synthetic_pairs)} pairs")

        # Merge: original + BM25-mined + synthetic
        original_rows = [row for row in dataset]
        all_rows = original_rows + bm25_pairs + synthetic_pairs
        log(f"{run_name}: merged dataset: {len(original_rows)} original + "
            f"{len(bm25_pairs)} BM25-mined + {len(synthetic_pairs)} synthetic "
            f"= {len(all_rows)} total pairs")
        dataset = HFDataset.from_list(all_rows)

        # Keep corpus for curriculum if needed, otherwise free
        if curriculum_easy_epochs > 0:
            curriculum_corpus = list(corpus_docs)  # copy before del
            log(f"{run_name}: curriculum enabled — keeping {len(curriculum_corpus)} docs for random negatives")
        # Free memory
        del corpus_docs, bm25_pairs, synthetic_pairs, original_rows
        import gc
        gc.collect()

    # --- Curriculum learning: load random negative pool if needed ---
    if curriculum_easy_epochs > 0 and not use_bm25_mining:
        # Load a sample of SAE corpus for random negatives
        SAE_CORPUS_CUR = f"{HF_USERNAME}/scar-corpus"
        log(f"{run_name}: curriculum — loading corpus sample for random negatives...")
        from datasets import load_dataset as _load_ds_cur
        cur_ds = _load_ds_cur(SAE_CORPUS_CUR, split="train", token=hf_token)
        cur_size = min(20000, len(cur_ds))
        cur_ds = cur_ds.select(range(cur_size))
        curriculum_corpus = [row["contract_code"] for row in cur_ds]
        del cur_ds
        log(f"{run_name}: curriculum corpus: {len(curriculum_corpus)} docs")

    # Add index column if distillation is active (needed to align scores with shuffled batches)
    use_distill = lambda_distill > 0
    if use_distill:
        dataset = dataset.map(lambda x, idx: {"_idx": idx}, with_indices=True)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset, batch_size=micro_batch_size, shuffle=True,
        collate_fn=create_collate_fn(return_indices=use_distill),
        num_workers=0, drop_last=True,
    )

    # --- Compute IDF weights ---
    # IDF down-weights universal features ("this is Solidity code") before TopK.
    retriever.eval()
    if idf_corpus_size > 0:
        # Corpus-level IDF: use SAE training corpus for more stable estimates
        SAE_CORPUS = f"{HF_USERNAME}/scar-corpus"
        log(f"{run_name}: loading {idf_corpus_size} docs from SAE corpus for IDF...")
        from datasets import load_dataset as _load_ds
        sae_ds = _load_ds(SAE_CORPUS, split="train", token=os.environ.get("HF_TOKEN"))
        sae_ds = sae_ds.select(range(min(idf_corpus_size, len(sae_ds))))
        idf_docs = [row["contract_code"] for row in sae_ds]
        del sae_ds
        idf_weights = retriever.compute_idf(idf_docs, batch_size=micro_batch_size)
        log(f"{run_name}: IDF from SAE corpus ({len(idf_docs)} docs)")
        del idf_docs
    else:
        # Default: IDF from training positive documents
        all_docs = [row["positive"] for row in dataset]
        idf_weights = retriever.compute_idf(all_docs, batch_size=micro_batch_size)
        log(f"{run_name}: IDF from training docs ({len(all_docs)} docs)")
    retriever.train()

    # --- Load cross-encoder distillation scores (v7) ---
    ce_pos_all = None
    ce_neg_all = None
    if lambda_distill > 0:
        scores_path = f"{SAVE_DIR}/cross_encoder/distill_scores.pt"
        if not os.path.exists(scores_path):
            log(f"WARNING: distillation requested but {scores_path} not found. Disabling.")
            lambda_distill = 0.0
        else:
            scores = torch.load(scores_path, map_location=device)
            ce_pos_all = scores["pos_scores"].to(device)
            ce_neg_all = scores["neg_scores"].to(device)
            log(f"{run_name}: loaded distillation scores ({ce_pos_all.shape[0]} pairs, "
                f"margin={( ce_pos_all - ce_neg_all).mean():.3f})")
            # Verify alignment with dataset
            if ce_pos_all.shape[0] != len(dataset):
                log(f"WARNING: score count ({ce_pos_all.shape[0]}) != dataset size ({len(dataset)}). "
                    f"Disabling distillation.")
                lambda_distill = 0.0
                ce_pos_all = None

    total_micro_steps_per_epoch = len(dataloader)
    total_opt_steps_per_epoch = total_micro_steps_per_epoch // grad_accum_steps
    total_opt_steps = total_opt_steps_per_epoch * epochs
    warmup_steps = int(total_opt_steps * warmup_ratio)

    log(f"{run_name}: {len(dataset)} pairs, {total_micro_steps_per_epoch} micro-batches/epoch, "
        f"{total_opt_steps} optimizer steps, {warmup_steps} warmup steps, "
        f"grad_accum={grad_accum_steps}")

    # --- Optimizer (LoRA params + SAE LoRA params if v5c) ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    sae_lora_params = []
    if sae_lora_rank > 0:
        sae_lora_params = [p for p in sae.parameters() if p.requires_grad]
        effective_sae_lr = sae_lora_lr if sae_lora_lr > 0 else lr
        log(f"{run_name}: optimizer includes {len(sae_lora_params)} SAE LoRA params "
            f"({sum(p.numel() for p in sae_lora_params):,} elements, lr={effective_sae_lr:.1e})")
        optimizer = torch.optim.AdamW([
            {"params": trainable_params, "lr": lr},
            {"params": sae_lora_params, "lr": effective_sae_lr},
        ], betas=(0.9, 0.999), weight_decay=0.01)
        for pg in optimizer.param_groups:
            pg["base_lr"] = pg["lr"]
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.999),
                                       weight_decay=0.01)
        for pg in optimizer.param_groups:
            pg["base_lr"] = pg["lr"]

    # Checkpoint targets (will remove already-hit ones after resume)
    ckpt_steps = {int(total_opt_steps * pct / 100): pct for pct in checkpoint_pcts}

    # --- Resume support ---
    resume_step = 0
    resume_epoch = 0
    for pct in sorted(checkpoint_pcts[:-1], reverse=True):
        ckpt_path = f"{run_dir}/checkpoint_{pct}pct.pt"
        if os.path.exists(ckpt_path):
            log(f"{run_name}: resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if has_peft:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(model, ckpt["model_state_dict"])
            if sae_lora_rank > 0 and "sae_lora_state" in ckpt:
                sae.load_state_dict(ckpt["sae_lora_state"], strict=False)
                log(f"{run_name}: SAE LoRA state restored from checkpoint")
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            resume_step = ckpt["step"]
            resume_epoch = ckpt.get("epoch", 0)
            log(f"{run_name}: resumed at step {resume_step}, epoch {resume_epoch}")
            # Remove checkpoint targets already passed
            ckpt_steps = {s: p for s, p in ckpt_steps.items() if s > resume_step}
            break

    # --- Temperature schedule helper ---
    def get_temperature(step, total_steps):
        """Get temperature for current step. Fixed or cosine cool-down."""
        if temp_schedule == "cosine_cooldown":
            progress = step / max(total_steps, 1)
            return temp_min + 0.5 * (temp_max - temp_min) * (1 + math.cos(math.pi * progress))
        return temperature  # fixed

    # --- GradCache helper ---
    def gradcache_step(all_queries, all_positives, all_hard_negatives, current_temp):
        """GradCache: compute contrastive loss over the full batch while
        only materializing one chunk at a time during backward.

        Algorithm (from Gao et al., 2021):
        1. Forward all chunks, collect sparse reps (detached)
        2. Compute full-batch contrastive loss on detached reps
        3. Backward through loss → get gradients on reps
        4. Re-forward each chunk and backward using cached rep gradients
        """
        chunk = gradcache_chunk_size
        n_chunks = math.ceil(len(all_queries) / chunk)

        # Phase 1: Forward all chunks, cache detached representations
        q_sparse_chunks, q_pooled_chunks = [], []
        p_sparse_chunks, p_pooled_chunks = [], []
        n_sparse_chunks, n_pooled_chunks = [], []

        for i in range(n_chunks):
            s, e = i * chunk, min((i + 1) * chunk, len(all_queries))
            with torch.no_grad():
                qs, qp = retriever.encode_queries(all_queries[s:e])
                ps, pp = retriever.encode_documents(all_positives[s:e])
                ns, np_ = retriever.encode_documents(all_hard_negatives[s:e])
            q_sparse_chunks.append(qs)
            q_pooled_chunks.append(qp)
            p_sparse_chunks.append(ps)
            p_pooled_chunks.append(pp)
            n_sparse_chunks.append(ns)
            n_pooled_chunks.append(np_)

        # Concatenate all detached reps
        all_q_sparse = torch.cat(q_sparse_chunks, dim=0).requires_grad_(True)
        all_p_sparse = torch.cat(p_sparse_chunks, dim=0).requires_grad_(True)
        all_n_sparse = torch.cat(n_sparse_chunks, dim=0)  # no grad for hard negs
        all_q_pooled = torch.cat(q_pooled_chunks, dim=0)
        all_p_pooled = torch.cat(p_pooled_chunks, dim=0)
        all_n_pooled = torch.cat(n_pooled_chunks, dim=0)

        # Phase 2: Full-batch contrastive loss
        loss, metrics = compute_total_loss(
            all_q_sparse, all_p_sparse, all_n_sparse,
            all_q_pooled, all_p_pooled, all_n_pooled,
            temperature=current_temp, lambda_d=lambda_d, lambda_q=lambda_q,
        )
        loss.backward()

        # Cache gradients on representations
        q_rep_grads = all_q_sparse.grad.detach().split([c.shape[0] for c in q_sparse_chunks])
        p_rep_grads = all_p_sparse.grad.detach().split([c.shape[0] for c in p_sparse_chunks])

        # Phase 3: Re-forward each chunk with grad, backward using cached rep grads
        for i in range(n_chunks):
            s, e = i * chunk, min((i + 1) * chunk, len(all_queries))

            qs, qp = retriever.encode_queries(all_queries[s:e])
            ps, pp = retriever.encode_documents(all_positives[s:e])

            # Surrogate loss: dot product of representation with cached gradient
            surrogate = (qs * q_rep_grads[i]).sum() + (ps * p_rep_grads[i]).sum()
            surrogate.backward()

        return loss.item(), metrics

    # --- Training loop ---
    opt_step = resume_step  # start from resume point (0 if fresh)
    training_log = []
    model.train()

    if use_gradcache:
        log(f"{run_name}: GradCache enabled (chunk_size={gradcache_chunk_size}, "
            f"effective_batch={batch_size})")
        # With GradCache, dataloader uses full batch_size
        dataloader = create_dataloader(dataset, batch_size)
        total_micro_steps_per_epoch = len(dataloader)
        total_opt_steps_per_epoch = total_micro_steps_per_epoch
        total_opt_steps = total_opt_steps_per_epoch * epochs
        warmup_steps = int(total_opt_steps * warmup_ratio)
        ckpt_steps = {int(total_opt_steps * pct / 100): pct for pct in checkpoint_pcts}
        grad_accum_steps = 1  # each dataloader batch = one optimizer step
        log(f"{run_name}: GradCache — {total_opt_steps} optimizer steps/run")

    if temp_schedule != "fixed":
        log(f"{run_name}: temperature schedule={temp_schedule}, "
            f"tau: {temp_max:.2f} → {temp_min:.2f}")

    if curriculum_easy_epochs > 0:
        log(f"{run_name}: curriculum learning — epochs 1-{curriculum_easy_epochs}: "
            f"random negatives, epochs {curriculum_easy_epochs + 1}-{epochs}: hard negatives")

    for epoch in range(epochs):
        if epoch < resume_epoch:
            continue

        use_easy_negatives = (curriculum_easy_epochs > 0 and epoch < curriculum_easy_epochs)
        neg_type = "random (easy)" if use_easy_negatives else "structured (hard)"
        log(f"{run_name}: epoch {epoch + 1}/{epochs} — negatives: {neg_type}")
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_metrics = {}

        for micro_idx, batch_data in enumerate(dataloader):
            # Unpack batch (with or without indices for distillation)
            if use_distill:
                queries, positives, hard_negatives, batch_indices = batch_data
            else:
                queries, positives, hard_negatives = batch_data
                batch_indices = None

            # Curriculum: replace hard negatives with random corpus docs during easy epochs
            if use_easy_negatives:
                hard_negatives = [
                    curriculum_corpus[random.randint(0, len(curriculum_corpus) - 1)]
                    for _ in range(len(hard_negatives))
                ]

            # Skip to resume point
            global_micro = epoch * total_micro_steps_per_epoch + micro_idx
            if global_micro < resume_step * grad_accum_steps:
                continue

            # Get current temperature
            current_opt_step = opt_step if (micro_idx + 1) % grad_accum_steps == 0 else opt_step
            current_temp = get_temperature(current_opt_step, total_opt_steps)

            # Get cross-encoder scores for this batch (if distilling)
            batch_ce_pos = None
            batch_ce_neg = None
            if use_distill and batch_indices is not None and ce_pos_all is not None:
                idx_tensor = torch.tensor(batch_indices, device=device)
                batch_ce_pos = ce_pos_all[idx_tensor]
                batch_ce_neg = ce_neg_all[idx_tensor]

            if use_gradcache:
                # GradCache path: full-batch contrastive loss with chunked backward
                loss_val, metrics = gradcache_step(
                    queries, positives, hard_negatives, current_temp)
                accum_loss = loss_val
                accum_metrics = metrics
            else:
                # Standard path: micro-batch forward + backward
                q_sparse, q_pooled = retriever.encode_queries(queries)
                p_sparse, p_pooled = retriever.encode_documents(positives)
                with torch.no_grad():
                    n_sparse, n_pooled = retriever.encode_documents(hard_negatives)

                loss, metrics = compute_total_loss(
                    q_sparse, p_sparse, n_sparse,
                    q_pooled, p_pooled, n_pooled,
                    temperature=current_temp, lambda_d=lambda_d, lambda_q=lambda_q,
                    ce_pos_scores=batch_ce_pos, ce_neg_scores=batch_ce_neg,
                    lambda_distill=lambda_distill,
                )

                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

                accum_loss += loss.item() / grad_accum_steps
                for k, v in metrics.items():
                    accum_metrics[k] = accum_metrics.get(k, 0) + v / grad_accum_steps

            # Optimizer step after accumulation
            if (micro_idx + 1) % grad_accum_steps == 0:
                all_graded = trainable_params + (sae_lora_params if sae_lora_rank > 0 else [])
                if all_graded:
                    torch.nn.utils.clip_grad_norm_(all_graded, grad_clip)

                # LR schedule: linear warmup then cosine annealing
                if opt_step < warmup_steps:
                    lr_mult = (opt_step + 1) / max(warmup_steps, 1)
                else:
                    progress = (opt_step - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
                    lr_mult = 0.5 * (1 + math.cos(math.pi * progress))
                current_lr = lr * lr_mult
                for pg in optimizer.param_groups:
                    pg["lr"] = pg.get("base_lr", lr) * lr_mult

                optimizer.step()
                optimizer.zero_grad()
                opt_step += 1

                # Logging
                if opt_step % log_every == 0:
                    accum_metrics["step"] = opt_step
                    accum_metrics["epoch"] = epoch + 1
                    accum_metrics["lr"] = current_lr
                    accum_metrics["temperature"] = current_temp

                    distill_str = ""
                    if "margin_mse" in accum_metrics:
                        distill_str = f" | mse={accum_metrics['margin_mse']:.4f}"
                    log(f"  step {opt_step}/{total_opt_steps} | "
                        f"loss={accum_metrics.get('total', 0):.4f} | "
                        f"infonce={accum_metrics.get('infonce', 0):.4f} | "
                        f"acc={accum_metrics.get('batch_acc', 0):.3f} | "
                        f"qL0={accum_metrics.get('query_l0', 0):.1f} | "
                        f"dL0={accum_metrics.get('doc_l0', 0):.1f} | "
                        f"lr={current_lr:.2e} | "
                        f"tau={current_temp:.3f}{distill_str}")

                    training_log.append(dict(accum_metrics))

                accum_loss = 0.0
                accum_metrics = {}

                # Checkpointing
                closest_ckpt = min(ckpt_steps.keys(), key=lambda s: abs(s - opt_step),
                                   default=None)
                if closest_ckpt is not None and abs(opt_step - closest_ckpt) < 2:
                    pct = ckpt_steps.pop(closest_ckpt)
                    ckpt_path = f"{run_dir}/checkpoint_{pct}pct.pt"
                    if not os.path.exists(ckpt_path):
                        sae_lora = {k: v for k, v in sae.state_dict().items()
                                    if k.startswith("lora_")} if sae_lora_rank > 0 else None
                        _save_retrieval_checkpoint(
                            ckpt_path, model, optimizer, config,
                            training_log[-1] if training_log else {},
                            opt_step, epoch, total_opt_steps,
                            idf_weights=retriever.idf_weights,
                            sae_lora_state=sae_lora,
                            has_peft=has_peft,
                        )
                        retrieval_vol.commit()
                        log(f"  checkpoint saved: {pct}%")

                # Memory cleanup
                if opt_step % 50 == 0:
                    torch.cuda.empty_cache()

    # --- Final save ---
    final_metrics = training_log[-1] if training_log else {}
    sae_lora_final = {k: v for k, v in sae.state_dict().items()
                      if k.startswith("lora_")} if sae_lora_rank > 0 else None
    _save_retrieval_checkpoint(
        final_ckpt, model, optimizer, config,
        final_metrics, opt_step, epochs, total_opt_steps,
        idf_weights=retriever.idf_weights,
        sae_lora_state=sae_lora_final,
        has_peft=has_peft,
    )

    # Save training log
    with open(f"{run_dir}/training_log.json", "w") as f:
        json.dump(training_log, f)

    # Save LoRA adapter in PEFT format (for clean Step 7 loading)
    if has_peft:
        lora_dir = f"{run_dir}/lora_adapter"
        model.save_pretrained(lora_dir)
        log(f"{run_name}: LoRA adapter saved to {lora_dir}")
    else:
        log(f"{run_name}: no backbone LoRA to save (SAE-LoRA only)")

    retrieval_vol.commit()

    log(f"{run_name}: TRAINING COMPLETE — {opt_step} steps, "
        f"acc={final_metrics.get('batch_acc', '?')}, "
        f"loss={final_metrics.get('total', '?')}")

    # --- Quick validation ---
    model.eval()
    try:
        val_metrics = quick_validate(retriever, hf_token=hf_token)
        log(f"{run_name}: validation — recall@10={val_metrics.get('recall_at_10', '?'):.3f}")
        final_metrics["val_recall_at_10"] = val_metrics.get("recall_at_10")
    except Exception as e:
        log(f"{run_name}: validation failed (non-fatal): {e}")

    # --- Push to HuggingFace ---
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        repo_id = f"{HF_USERNAME}/scar-weights"
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True,
                        token=hf_token)
        api.upload_file(
            path_or_fileobj=final_ckpt,
            path_in_repo=f"retrieval/{run_name}/checkpoint_final.pt",
            repo_id=repo_id, repo_type="model", token=hf_token,
            commit_message=f"Upload retrieval checkpoint: {run_name}",
        )
        api.upload_file(
            path_or_fileobj=f"{run_dir}/config.json",
            path_in_repo=f"retrieval/{run_name}/config.json",
            repo_id=repo_id, repo_type="model", token=hf_token,
        )
        api.upload_folder(
            folder_path=lora_dir,
            path_in_repo=f"retrieval/{run_name}/lora_adapter",
            repo_id=repo_id, repo_type="model", token=hf_token,
        )
        log(f"{run_name}: pushed to {repo_id}")
    except Exception as e:
        log(f"{run_name}: HF push failed (non-fatal): {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Temperature Grid Search
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={
        SAE_DIR: sae_vol,
        SAVE_DIR: retrieval_vol,
    },
    secrets=[HF_SECRET],
    timeout=14400,
)
def temperature_sweep(
    candidates: list = None,
    sweep_steps: int = 1000,
    micro_batch_size: int = 16,
    max_seq_len: int = 256,
    lr: float = 5e-5,
    lora_rank: int = 64,
    topk_query: int = 40,
    topk_doc: int = 400,
    sae_run_name: str = "primary_L19_16k",
    force: bool = False,
    lambda_d: float = 0.0,
    lambda_q: float = 0.0,
):
    """Quick training runs at different temperatures to find optimal tau.

    SPLARE found temperature is the MOST SENSITIVE hyperparameter.
    With L2-normalized sparse vectors (cosine similarity, bounded [-1,1]),
    temperatures should be in [0.01, 1.0] range (SimCLR uses 0.07-0.5).
    Saves results to temperature_sweep/results.json.
    """
    import torch
    import json
    import shutil

    if candidates is None:
        candidates = [0.02, 0.05, 0.1, 0.5]

    sae_vol.reload()
    retrieval_vol.reload()

    sweep_dir = f"{SAVE_DIR}/temperature_sweep"
    output_path = f"{sweep_dir}/results.json"

    # Force re-run: delete stale results from previous failed runs
    if force and os.path.exists(sweep_dir):
        shutil.rmtree(sweep_dir)
        log("Temperature sweep: force=True, deleted stale results")

    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        all_covered = all(str(tau) in existing for tau in candidates)
        if all_covered and "best_temperature" in existing:
            log("Temperature sweep: already done, skipping")
            log(f"Temperature results: {json.dumps(existing, indent=2)}")
            return existing["best_temperature"]
        log("Temperature sweep: re-running with new candidates")

    os.makedirs(sweep_dir, exist_ok=True)

    device = torch.device("cuda")
    hf_token = os.environ.get("HF_TOKEN")

    # Load SAE once (shared across candidates)
    sae_ckpt_path = f"{SAE_DIR}/{sae_run_name}/checkpoint_final.pt"
    if not os.path.exists(sae_ckpt_path):
        log(f"Temperature sweep: ERROR — SAE checkpoint not found at {sae_ckpt_path}")
        return None

    # Load dataset once
    dataset = load_contrastive_dataset(hf_token=hf_token)

    results = {}

    for tau in candidates:
        log(f"Sweep: testing temperature={tau}")

        # Full model setup for each candidate (fresh LoRA weights each time)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig, TaskType

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device)

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(model, lora_config)
        # Bidirectional attention: JumpReLU bypassed via per-token TopK
        enable_bidirectional_attention(model)

        sae, target_norm = load_frozen_sae(sae_ckpt_path, device)

        RetrieverClass = build_retriever_class()
        retriever = RetrieverClass(
            model, tokenizer, sae, layer_idx=LAYER_IDX,
            topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
            target_norm=target_norm,
        )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.999),
                                       weight_decay=0.01)

        dataloader = create_dataloader(dataset, micro_batch_size)
        model.train()

        step = 0
        loss_history = []
        acc_history = []
        tail_start = int(sweep_steps * 0.8)

        # Cycle through dataloader for multiple epochs (dataset has ~252 batches)
        import itertools
        for queries, positives, hard_negatives in itertools.chain.from_iterable(
            itertools.repeat(dataloader)
        ):
            q_sparse, q_pooled = retriever.encode_queries(queries)
            p_sparse, p_pooled = retriever.encode_documents(positives)
            # Hard negatives: no grad to save memory (gradient flows through q and p)
            with torch.no_grad():
                n_sparse, n_pooled = retriever.encode_documents(hard_negatives)

            loss, metrics = compute_total_loss(
                q_sparse, p_sparse, n_sparse,
                q_pooled, p_pooled, n_pooled,
                temperature=tau,
                lambda_d=lambda_d, lambda_q=lambda_q,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1

            if step % 30 == 0:
                log(f"  tau={tau} step {step}/{sweep_steps}: "
                    f"total={metrics['total']:.4f} infonce={metrics['infonce']:.4f} "
                    f"df_doc={metrics['df_flops_doc']:.1f} df_q={metrics['df_flops_query']:.1f} "
                    f"acc={metrics['batch_acc']:.3f}")

            if step > tail_start:
                loss_history.append(metrics["infonce"])
                acc_history.append(metrics["batch_acc"])

            if step >= sweep_steps:
                break

        avg_loss = sum(loss_history) / max(len(loss_history), 1)
        avg_acc = sum(acc_history) / max(len(acc_history), 1)
        results[str(tau)] = {
            "avg_infonce": round(avg_loss, 4),
            "avg_acc": round(avg_acc, 4),
            "steps": step,
        }
        log(f"  tau={tau} → infonce={avg_loss:.4f}, acc={avg_acc:.4f}")

        # Cleanup for next candidate
        del model, optimizer, retriever, sae, trainable_params, tokenizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Select best (lowest InfoNCE loss)
    best_tau = min(candidates, key=lambda t: results[str(t)]["avg_infonce"])
    results["best_temperature"] = best_tau

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    retrieval_vol.commit()

    log(f"Temperature sweep complete. Best tau={best_tau} "
        f"(infonce={results[str(best_tau)]['avg_infonce']})")

    return best_tau


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: Quick Validation
# ═══════════════════════════════════════════════════════════════════════════

def quick_validate(retriever, hf_token=None, max_eval=200):
    """Quick Recall@10 on FORGE-Curated eval holdout.

    NOT the full evaluation (that's Step 9). This is a sanity check that
    the model is learning to retrieve.

    Args:
        retriever: ScarRetriever instance (in eval mode)
        hf_token: HuggingFace token
        max_eval: max pairs to evaluate (for speed)

    Returns:
        dict with recall_at_10
    """
    import torch
    from datasets import load_dataset

    ds = load_dataset(EVAL_DATASET, split="train", token=hf_token)
    if max_eval and len(ds) > max_eval:
        ds = ds.select(range(max_eval))

    log(f"Validation: encoding {len(ds)} eval pairs...")

    # Collect all queries and positives
    queries = [row["query"] for row in ds]
    positives = [row["ground_truth_code"] for row in ds]

    # Encode in batches (no gradient)
    batch_size = 16
    all_q_vecs = []
    all_d_vecs = []

    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            q_batch = queries[i:i + batch_size]
            d_batch = positives[i:i + batch_size]
            q_sparse, _ = retriever.encode_queries(q_batch)
            d_sparse, _ = retriever.encode_documents(d_batch)
            all_q_vecs.append(q_sparse.cpu())
            all_d_vecs.append(d_sparse.cpu())

    q_matrix = torch.cat(all_q_vecs, dim=0)  # (N, d_sae)
    d_matrix = torch.cat(all_d_vecs, dim=0)  # (N, d_sae)

    # Compute full similarity matrix and Recall@10
    sim = torch.matmul(q_matrix, d_matrix.T)  # (N, N)
    _, top_indices = sim.topk(10, dim=1)  # (N, 10)

    correct = torch.arange(len(queries)).unsqueeze(1)  # (N, 1)
    hits = (top_indices == correct).any(dim=1).float()
    recall_at_10 = hits.mean().item()

    log(f"Validation: recall@10 = {recall_at_10:.3f} ({hits.sum().int().item()}/{len(queries)})")

    return {"recall_at_10": recall_at_10, "n_eval": len(queries)}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9b: Diagnostic Analysis
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
def diagnose_retrieval(
    topk_query: int = 100,
    topk_doc: int = 400,
    max_seq_len: int = 256,
    lora_rank: int = 64,
    sae_run_name: str = "primary_L19_16k",
    use_lora: bool = True,
    use_bidirectional: bool = True,
    use_idf: bool = True,
):
    """Deep diagnostic of retrieval representations.

    Analyzes: similarity matrix, feature diversity, representation collapse,
    and compares trained vs untrained models.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sae_vol.reload()
    retrieval_vol.reload()

    device = torch.device("cuda")
    log(f"[DIAG] Starting diagnostic (use_lora={use_lora}, use_bidirectional={use_bidirectional})")
    log(f"[DIAG] GPU: {torch.cuda.get_device_name(0)}")

    # --- Load model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager",
    )
    log(f"[DIAG] Model loaded")

    # Optionally apply LoRA
    if use_lora:
        lora_dir = f"{SAVE_DIR}/primary/lora_adapter"
        if os.path.exists(lora_dir):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_dir)
            log(f"[DIAG] LoRA loaded from {lora_dir}")
        else:
            log(f"[DIAG] WARNING: LoRA dir {lora_dir} not found, running without LoRA")
            use_lora = False

    model = model.to(device).eval()

    # Optionally enable bidirectional attention
    if use_bidirectional:
        enable_bidirectional_attention(model)

    # Load SAE
    sae_ckpt = f"{SAE_DIR}/{sae_run_name}/checkpoint_final.pt"
    sae, target_norm = load_frozen_sae(sae_ckpt, device)
    log(f"[DIAG] SAE loaded (d_sae={sae.d_sae})")

    # Build retriever
    RetrieverClass = build_retriever_class()
    retriever = RetrieverClass(
        model, tokenizer, sae, layer_idx=LAYER_IDX,
        topk_query=topk_query, topk_doc=topk_doc, max_seq_len=max_seq_len,
        target_norm=target_norm,
    )

    # Compute IDF from training corpus
    hf_token = os.environ.get("HF_TOKEN")
    if use_idf:
        train_ds = load_dataset(PAIRS_DATASET, split="train", token=hf_token)
        all_train_docs = [row["positive"] for row in train_ds]
        retriever.compute_idf(all_train_docs, batch_size=16)
        log(f"[DIAG] IDF computed from {len(all_train_docs)} training documents")
    else:
        log(f"[DIAG] IDF disabled")

    # Load eval dataset
    ds = load_dataset(EVAL_DATASET, split="train", token=hf_token)
    queries = [row["query"] for row in ds]
    positives = [row["ground_truth_code"] for row in ds]
    N = len(queries)
    log(f"[DIAG] Eval dataset: {N} pairs")

    # Print sample data
    log(f"[DIAG] Sample query (first 200 chars): {queries[0][:200]}")
    log(f"[DIAG] Sample code (first 200 chars): {positives[0][:200]}")

    # Encode all queries and documents
    all_q = []
    all_d = []
    all_q_pooled = []
    all_d_pooled = []

    batch_size = 8
    with torch.no_grad():
        for i in range(0, N, batch_size):
            q_batch = queries[i:i+batch_size]
            d_batch = positives[i:i+batch_size]
            q_sparse, q_pool = retriever.encode_queries(q_batch)
            d_sparse, d_pool = retriever.encode_documents(d_batch)
            all_q.append(q_sparse.cpu())
            all_d.append(d_sparse.cpu())
            all_q_pooled.append(q_pool.cpu())
            all_d_pooled.append(d_pool.cpu())

    Q = torch.cat(all_q, dim=0)      # (N, d_sae) L2-normed sparse
    D = torch.cat(all_d, dim=0)      # (N, d_sae) L2-normed sparse
    Q_pool = torch.cat(all_q_pooled, dim=0)  # pre-TopK pooled
    D_pool = torch.cat(all_d_pooled, dim=0)

    # === 1. Similarity Matrix Analysis ===
    sim = torch.matmul(Q, D.T)  # (N, N) cosine similarities
    diag = sim.diag()  # correct pair similarities

    log(f"\n{'='*60}")
    log(f"[DIAG] SIMILARITY MATRIX (Q @ D^T, {N}x{N})")
    log(f"  Overall: mean={sim.mean():.4f}, std={sim.std():.4f}, min={sim.min():.4f}, max={sim.max():.4f}")
    log(f"  Diagonal (correct pairs): mean={diag.mean():.4f}, std={diag.std():.4f}, min={diag.min():.4f}, max={diag.max():.4f}")

    # Off-diagonal (wrong pairs)
    mask = ~torch.eye(N, dtype=torch.bool)
    off_diag = sim[mask]
    log(f"  Off-diagonal (wrong pairs): mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
    log(f"  Margin (diag - off_diag mean): {(diag.mean() - off_diag.mean()):.4f}")

    # Separation
    correct_better = 0
    for i in range(N):
        row = sim[i]
        if row[i] > row.median():
            correct_better += 1
    log(f"  Correct pair > median of row: {correct_better}/{N} ({correct_better/N:.1%})")

    # R@K for different K
    for k in [1, 5, 10, 20, 50]:
        _, top_idx = sim.topk(k, dim=1)
        correct = torch.arange(N).unsqueeze(1)
        hits = (top_idx == correct).any(dim=1).float().mean().item()
        log(f"  R@{k}: {hits:.3f} ({int(hits*N)}/{N})")

    # === 2. Feature Diversity ===
    log(f"\n{'='*60}")
    log(f"[DIAG] FEATURE DIVERSITY")

    q_active = (Q > 0).float()
    d_active = (D > 0).float()

    q_l0 = q_active.sum(dim=1)
    d_l0 = d_active.sum(dim=1)
    log(f"  Query L0: mean={q_l0.mean():.1f}, std={q_l0.std():.1f}, min={q_l0.min():.0f}, max={q_l0.max():.0f}")
    log(f"  Doc L0: mean={d_l0.mean():.1f}, std={d_l0.std():.1f}, min={d_l0.min():.0f}, max={d_l0.max():.0f}")

    # How many unique features are active across ALL queries/docs?
    any_q_active = (q_active.sum(dim=0) > 0).sum().item()
    any_d_active = (d_active.sum(dim=0) > 0).sum().item()
    log(f"  Unique features active across queries: {any_q_active}/{Q.shape[1]}")
    log(f"  Unique features active across docs: {any_d_active}/{D.shape[1]}")

    # Feature frequency: how many docs share each active feature?
    feat_freq_d = d_active.sum(dim=0)  # (d_sae,) — count of docs using each feature
    active_feats = feat_freq_d[feat_freq_d > 0]
    log(f"  Doc feature frequency: mean={active_feats.mean():.1f}, median={active_feats.median():.1f}, "
        f"max={active_feats.max():.0f}")

    # How many features are shared by >50% of docs? (too common = not discriminative)
    common_feats = (feat_freq_d > N * 0.5).sum().item()
    very_common = (feat_freq_d > N * 0.8).sum().item()
    log(f"  Features in >50% of docs: {common_feats}")
    log(f"  Features in >80% of docs: {very_common}")

    # === 3. Feature Overlap Analysis ===
    log(f"\n{'='*60}")
    log(f"[DIAG] FEATURE OVERLAP")

    # Overlap between correct query-doc pairs
    correct_overlaps = []
    random_overlaps = []
    for i in range(N):
        q_feats = set((Q[i] > 0).nonzero(as_tuple=True)[0].tolist())
        d_feats = set((D[i] > 0).nonzero(as_tuple=True)[0].tolist())
        correct_overlaps.append(len(q_feats & d_feats))
        # Random pair
        j = (i + 1) % N
        d_feats_rand = set((D[j] > 0).nonzero(as_tuple=True)[0].tolist())
        random_overlaps.append(len(q_feats & d_feats_rand))

    correct_overlaps = np.array(correct_overlaps)
    random_overlaps = np.array(random_overlaps)
    log(f"  Correct pair overlap: mean={correct_overlaps.mean():.1f}, std={correct_overlaps.std():.1f}")
    log(f"  Random pair overlap: mean={random_overlaps.mean():.1f}, std={random_overlaps.std():.1f}")
    log(f"  Overlap ratio (correct/random): {correct_overlaps.mean()/max(random_overlaps.mean(), 0.001):.2f}x")

    # === 4. Pairwise Doc Similarity ===
    log(f"\n{'='*60}")
    log(f"[DIAG] DOCUMENT PAIRWISE SIMILARITY")
    dd_sim = torch.matmul(D, D.T)
    dd_mask = ~torch.eye(N, dtype=torch.bool)
    dd_off = dd_sim[dd_mask]
    log(f"  D@D^T off-diagonal: mean={dd_off.mean():.4f}, std={dd_off.std():.4f}, max={dd_off.max():.4f}")

    qq_sim = torch.matmul(Q, Q.T)
    qq_off = qq_sim[dd_mask]
    log(f"  Q@Q^T off-diagonal: mean={qq_off.mean():.4f}, std={qq_off.std():.4f}, max={qq_off.max():.4f}")

    # === 5. Pre-TopK Pooled Analysis (shows what features are available before sparsification) ===
    log(f"\n{'='*60}")
    log(f"[DIAG] PRE-TOPK POOLED VALUES")
    q_pool_l0 = (Q_pool > 0).float().sum(dim=1)
    d_pool_l0 = (D_pool > 0).float().sum(dim=1)
    log(f"  Query pre-TopK L0: mean={q_pool_l0.mean():.0f}")
    log(f"  Doc pre-TopK L0: mean={d_pool_l0.mean():.0f}")
    log(f"  Query pooled max: {Q_pool.max():.2f}, mean_active: {Q_pool[Q_pool>0].mean():.4f}")
    log(f"  Doc pooled max: {D_pool.max():.2f}, mean_active: {D_pool[D_pool>0].mean():.4f}")

    # === 6. Token-level analysis (encode one pair and inspect) ===
    log(f"\n{'='*60}")
    log(f"[DIAG] SAMPLE PAIR DEEP DIVE (pair 0)")
    log(f"  Query: {queries[0][:100]}...")
    log(f"  Doc: {positives[0][:100]}...")
    log(f"  Q-D similarity: {sim[0, 0]:.4f}")
    log(f"  Q overlap with correct D: {correct_overlaps[0]} features")
    log(f"  Q overlap with random D: {random_overlaps[0]} features")
    log(f"  Top-5 most similar docs to query 0:")
    top5_sim, top5_idx = sim[0].topk(5)
    for rank, (s, idx) in enumerate(zip(top5_sim, top5_idx)):
        is_correct = "✓" if idx.item() == 0 else "✗"
        log(f"    Rank {rank+1}: idx={idx.item()} sim={s:.4f} {is_correct}")

    log(f"\n{'='*60}")
    log(f"[DIAG] SUMMARY")
    log(f"  Margin = {(diag.mean() - off_diag.mean()):.4f}")
    log(f"  Feature overlap ratio = {correct_overlaps.mean()/max(random_overlaps.mean(), 0.001):.2f}x")
    if diag.mean() - off_diag.mean() < 0.01:
        log(f"  ⚠️  NEAR-ZERO MARGIN: representations are NOT discriminative")
        log(f"     Likely causes: SAE features not semantic, representation collapse")
    elif correct_overlaps.mean() / max(random_overlaps.mean(), 0.001) < 1.2:
        log(f"  ⚠️  LOW OVERLAP RATIO: correct pairs don't share more features than random")
        log(f"     Likely cause: TopK selecting common features, not discriminative ones")
    else:
        log(f"  ✓ Representations show some discrimination")

    return {
        "r_at_10": ((sim.topk(10, dim=1)[1] == torch.arange(N).unsqueeze(1)).any(dim=1).float().mean().item()),
        "margin": (diag.mean() - off_diag.mean()).item(),
        "overlap_ratio": float(correct_overlaps.mean() / max(random_overlaps.mean(), 0.001)),
        "dd_mean_sim": dd_off.mean().item(),
        "qq_mean_sim": qq_off.mean().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9b: SAE Feature Interpretability Analysis
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
def analyze_sae_features(
    sae_run_name: str = "primary_L19_16k",
    max_seq_len: int = 256,
    n_corpus_samples: int = 2000,
    n_top_features: int = 50,
    n_top_tokens_per_feature: int = 20,
):
    """Interpretability analysis of SAE features on smart contract code.

    This is the critical diagnostic: do SAE features capture meaningful
    security-relevant patterns, or just generic code structure?

    Analysis:
    1. Pass 2000 contracts through model → SAE, get per-token feature activations
    2. For each of the top-50 most active features:
       - Find the tokens that maximally activate it
       - Cluster by token context (what code pattern surrounds the token?)
       - Label: security-relevant vs generic
    3. Pass 10 vulnerability queries, check which features activate
    4. Compare query features vs document features (modality gap analysis)
    """
    import torch
    import json
    import numpy as np
    from collections import defaultdict, Counter
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sae_vol.reload()
    device = torch.device("cuda")
    log("[INTERP] Starting SAE feature interpretability analysis")
    log(f"[INTERP] GPU: {torch.cuda.get_device_name(0)}")

    # --- Load model (no LoRA needed — we're analyzing raw SAE features) ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # --- Load SAE ---
    sae_ckpt_path = f"{SAE_DIR}/{sae_run_name}/checkpoint_final.pt"
    sae, target_norm = load_frozen_sae(sae_ckpt_path, device)
    log(f"[INTERP] SAE loaded: d_sae={sae.d_sae}, target_norm={target_norm:.2f}")

    # --- Hook to capture Layer 19 activations ---
    _activations = {}
    target_layer = _get_target_layer(model, LAYER_IDX)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            _activations["hidden"] = output[0]
        else:
            _activations["hidden"] = output

    hook = target_layer.register_forward_hook(hook_fn)

    # --- Load corpus sample ---
    hf_token = os.environ.get("HF_TOKEN")
    from datasets import load_dataset
    log(f"[INTERP] Loading {n_corpus_samples} contracts from SAE corpus...")
    sae_ds = load_dataset(f"{HF_USERNAME}/scar-corpus", split="train", token=hf_token)
    sae_ds = sae_ds.shuffle(seed=42).select(range(min(n_corpus_samples, len(sae_ds))))
    corpus_texts = [row["contract_code"] for row in sae_ds]
    del sae_ds

    # --- Load eval queries ---
    eval_ds = load_dataset(f"{HF_USERNAME}/scar-eval", split="train", token=hf_token)
    eval_queries = [row["query"] for row in eval_ds][:20]
    eval_codes = [row["ground_truth_code"] for row in eval_ds][:20]
    log(f"[INTERP] Loaded {len(eval_queries)} eval queries")

    # ====================================================================
    # PART 1: Per-feature analysis on corpus
    # For each feature, collect: (token_text, context_window, activation_value)
    # ====================================================================
    log("[INTERP] PART 1: Analyzing feature activations on corpus...")

    # feature_id → list of (token_str, left_context, right_context, activation_value)
    feature_examples = defaultdict(list)
    # feature_id → total activation across all tokens
    feature_total_activation = np.zeros(sae.d_sae)
    # feature_id → count of tokens where it appears in top-64
    feature_token_count = np.zeros(sae.d_sae)
    total_tokens_processed = 0

    batch_size = 8
    for batch_start in range(0, len(corpus_texts), batch_size):
        batch_texts = corpus_texts[batch_start:batch_start + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_len,
        ).to(device)

        with torch.no_grad():
            model(**inputs)
            hidden = _activations["hidden"]  # (B, T, 1536)

            # Normalize to SAE training scale
            norms = hidden.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            hidden_normed = hidden * (target_norm / norms)

            B, T, D = hidden_normed.shape
            flat = hidden_normed.reshape(-1, D)
            _, z_pre = sae.encode(flat)  # (B*T, 16384)

            # Per-token TopK(64)
            topk_vals, topk_idx = torch.topk(z_pre, 64, dim=-1)
            topk_vals = torch.relu(topk_vals)

            # Collect per-feature stats
            attention_mask = inputs["attention_mask"]  # (B, T)
            input_ids = inputs["input_ids"]  # (B, T)

            for b in range(B):
                tokens = tokenizer.convert_ids_to_tokens(input_ids[b])
                for t in range(T):
                    if attention_mask[b, t] == 0:
                        continue
                    total_tokens_processed += 1
                    token_idx = b * T + t

                    for k in range(64):
                        feat_id = topk_idx[token_idx, k].item()
                        val = topk_vals[token_idx, k].item()
                        if val <= 0:
                            continue

                        feature_total_activation[feat_id] += val
                        feature_token_count[feat_id] += 1

                        # Store example (keep top examples per feature)
                        if len(feature_examples[feat_id]) < n_top_tokens_per_feature:
                            left_ctx = " ".join(tokens[max(0, t-5):t])
                            right_ctx = " ".join(tokens[t+1:min(T, t+6)])
                            tok_str = tokens[t] if t < len(tokens) else "?"
                            feature_examples[feat_id].append({
                                "token": tok_str,
                                "left": left_ctx,
                                "right": right_ctx,
                                "activation": round(val, 3),
                            })
                        elif val > min(e["activation"] for e in feature_examples[feat_id]):
                            # Replace weakest example
                            min_idx = min(range(len(feature_examples[feat_id])),
                                          key=lambda i: feature_examples[feat_id][i]["activation"])
                            left_ctx = " ".join(tokens[max(0, t-5):t])
                            right_ctx = " ".join(tokens[t+1:min(T, t+6)])
                            tok_str = tokens[t] if t < len(tokens) else "?"
                            feature_examples[feat_id][min_idx] = {
                                "token": tok_str,
                                "left": left_ctx,
                                "right": right_ctx,
                                "activation": round(val, 3),
                            }

        if (batch_start + batch_size) % 200 == 0:
            log(f"  Processed {batch_start + batch_size}/{len(corpus_texts)} contracts, "
                f"{total_tokens_processed} tokens")

    log(f"[INTERP] Corpus analysis complete: {total_tokens_processed} tokens from "
        f"{len(corpus_texts)} contracts")

    # --- Find top features by total activation ---
    active_features = np.where(feature_token_count > 0)[0]
    log(f"[INTERP] Active features: {len(active_features)}/{sae.d_sae} "
        f"({len(active_features)/sae.d_sae*100:.1f}%)")

    top_feature_ids = feature_total_activation.argsort()[::-1][:n_top_features]

    log(f"\n[INTERP] ═══ TOP {n_top_features} SAE FEATURES (by total activation) ═══")
    feature_report = []
    for rank, feat_id in enumerate(top_feature_ids):
        examples = feature_examples.get(feat_id, [])
        examples.sort(key=lambda e: -e["activation"])

        # Analyze token patterns
        token_counter = Counter(e["token"] for e in examples)
        top_tokens = token_counter.most_common(5)

        # Check for security-relevant patterns
        security_keywords = [
            'transfer', 'call', 'delegatecall', 'send', 'selfdestruct',
            'msg.sender', 'owner', 'admin', 'require', 'assert',
            'balance', 'withdraw', 'deposit', 'approve', 'allowance',
            'reentrancy', 'overflow', 'underflow', 'external', 'payable',
            'modifier', 'onlyOwner', 'auth', 'access', 'lock', 'guard',
        ]
        all_context = " ".join(e["left"] + " " + e["token"] + " " + e["right"]
                               for e in examples).lower()
        sec_hits = [kw for kw in security_keywords if kw.lower() in all_context]

        feat_info = {
            "rank": rank + 1,
            "feature_id": int(feat_id),
            "total_activation": round(float(feature_total_activation[feat_id]), 2),
            "token_count": int(feature_token_count[feat_id]),
            "top_tokens": top_tokens,
            "security_keywords_in_context": sec_hits,
            "examples": examples[:10],
        }
        feature_report.append(feat_info)

        # Print summary
        tok_summary = ", ".join(f"'{t}'({c})" for t, c in top_tokens[:3])
        sec_label = f" SEC:[{','.join(sec_hits[:3])}]" if sec_hits else ""
        log(f"  #{rank+1} Feature {feat_id}: "
            f"total_act={feature_total_activation[feat_id]:.1f}, "
            f"count={feature_token_count[feat_id]:.0f}, "
            f"tokens=[{tok_summary}]{sec_label}")

        # Print top 3 examples with context
        for ex in examples[:3]:
            log(f"      [{ex['activation']:.2f}] ...{ex['left']} >>>{ex['token']}<<< {ex['right']}...")

    # ====================================================================
    # PART 2: Query vs Document feature analysis (modality gap)
    # ====================================================================
    log(f"\n[INTERP] ═══ PART 2: QUERY vs DOCUMENT MODALITY GAP ═══")

    def get_feature_activations(texts, label):
        """Get per-feature activation stats for a set of texts."""
        feat_acts = np.zeros(sae.d_sae)
        feat_counts = np.zeros(sae.d_sae)
        n_tokens = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_seq_len,
            ).to(device)

            with torch.no_grad():
                model(**inputs)
                hidden = _activations["hidden"]
                norms = hidden.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                hidden_normed = hidden * (target_norm / norms)

                flat = hidden_normed.reshape(-1, hidden_normed.shape[-1])
                _, z_pre = sae.encode(flat)
                topk_vals, topk_idx = torch.topk(z_pre, 64, dim=-1)
                topk_vals = torch.relu(topk_vals)

                mask = inputs["attention_mask"].reshape(-1)
                for tok_idx in range(flat.shape[0]):
                    if mask[tok_idx] == 0:
                        continue
                    n_tokens += 1
                    for k in range(64):
                        fid = topk_idx[tok_idx, k].item()
                        val = topk_vals[tok_idx, k].item()
                        if val > 0:
                            feat_acts[fid] += val
                            feat_counts[fid] += 1

        log(f"  [{label}] {n_tokens} tokens, "
            f"{int((feat_counts > 0).sum())} active features")
        return feat_acts, feat_counts, n_tokens

    q_acts, q_counts, q_tokens = get_feature_activations(eval_queries, "QUERIES")
    d_acts, d_counts, d_tokens = get_feature_activations(eval_codes[:20], "DOCUMENTS")

    # Features unique to queries vs documents
    q_only = set(np.where((q_counts > 0) & (d_counts == 0))[0])
    d_only = set(np.where((d_counts > 0) & (q_counts == 0))[0])
    shared = set(np.where((q_counts > 0) & (d_counts > 0))[0])

    log(f"\n  Feature overlap:")
    log(f"    Query-only features:    {len(q_only)}")
    log(f"    Document-only features: {len(d_only)}")
    log(f"    Shared features:        {len(shared)}")
    log(f"    Jaccard similarity:     {len(shared)/max(len(q_only|d_only|shared),1):.3f}")

    # Top query-distinctive features (high in queries, low in docs)
    q_norm = q_acts / max(q_tokens, 1)
    d_norm = d_acts / max(d_tokens, 1)
    q_distinctive = (q_norm - d_norm)
    top_q_distinctive = q_distinctive.argsort()[::-1][:10]

    log(f"\n  Top 10 QUERY-DISTINCTIVE features (active in queries, not docs):")
    for feat_id in top_q_distinctive:
        examples = feature_examples.get(feat_id, [])
        tok_summary = ", ".join(f"'{e['token']}'" for e in examples[:3]) if examples else "no examples"
        log(f"    Feature {feat_id}: q_act={q_norm[feat_id]:.3f}, d_act={d_norm[feat_id]:.3f}, "
            f"tokens=[{tok_summary}]")

    # Top document-distinctive features
    d_distinctive = (d_norm - q_norm)
    top_d_distinctive = d_distinctive.argsort()[::-1][:10]

    log(f"\n  Top 10 DOCUMENT-DISTINCTIVE features (active in docs, not queries):")
    for feat_id in top_d_distinctive:
        examples = feature_examples.get(feat_id, [])
        tok_summary = ", ".join(f"'{e['token']}'" for e in examples[:3]) if examples else "no examples"
        log(f"    Feature {feat_id}: q_act={q_norm[feat_id]:.3f}, d_act={d_norm[feat_id]:.3f}, "
            f"tokens=[{tok_summary}]")

    # ====================================================================
    # PART 3: Per-query feature analysis
    # ====================================================================
    log(f"\n[INTERP] ═══ PART 3: PER-QUERY FEATURE ANALYSIS ═══")

    for qi in range(min(10, len(eval_queries))):
        query = eval_queries[qi]
        query_short = query[:80] + "..." if len(query) > 80 else query

        inputs = tokenizer(
            [query], return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_len,
        ).to(device)

        with torch.no_grad():
            model(**inputs)
            hidden = _activations["hidden"]
            norms = hidden.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            hidden_normed = hidden * (target_norm / norms)

            flat = hidden_normed.reshape(-1, hidden_normed.shape[-1])
            _, z_pre = sae.encode(flat)
            topk_vals, topk_idx = torch.topk(z_pre, 64, dim=-1)
            topk_vals = torch.relu(topk_vals)

            # Pool: sum over tokens → doc-level TopK(100)
            pooled = torch.zeros(sae.d_sae, device=device)
            mask = inputs["attention_mask"][0]
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            for t in range(flat.shape[0]):
                if mask[t] == 0:
                    continue
                for k in range(64):
                    fid = topk_idx[t, k].item()
                    val = topk_vals[t, k].item()
                    if val > 0:
                        pooled[fid] += val

            # Top-10 features for this query
            top_feats = pooled.topk(10)

        log(f"\n  Query {qi+1}: {query_short}")
        for rank in range(10):
            fid = top_feats.indices[rank].item()
            val = top_feats.values[rank].item()
            examples = feature_examples.get(fid, [])
            tok_summary = ", ".join(f"'{e['token']}'" for e in examples[:3]) if examples else "?"
            log(f"    Feature {fid} (act={val:.2f}): typical tokens=[{tok_summary}]")

    # ====================================================================
    # Save report
    # ====================================================================
    hook.remove()

    report = {
        "total_tokens_processed": total_tokens_processed,
        "total_active_features": int(len(active_features)),
        "total_sae_features": sae.d_sae,
        "feature_utilization": round(len(active_features) / sae.d_sae, 4),
        "modality_gap": {
            "query_only_features": len(q_only),
            "doc_only_features": len(d_only),
            "shared_features": len(shared),
            "jaccard": round(len(shared)/max(len(q_only|d_only|shared),1), 4),
        },
        "top_features": feature_report,
    }

    report_path = f"{SAVE_DIR}/interpretability_report.json"
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    retrieval_vol.commit()

    log(f"\n[INTERP] Report saved to {report_path}")
    log("[INTERP] ANALYSIS COMPLETE")

    return report


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9b: IDF-Based Feature Interpretability (trained model)
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
def analyze_sae_features_idf(
    run_name: str = "v13_e25",
    sae_run_name: str = "primary_L19_16k",
    sae_lora_rank: int = 256,
    lora_rank: int = 64,
    max_seq_len: int = 256,
    n_corpus_samples: int = 2000,
    n_top_features: int = 10,
    n_top_docs: int = 5,
):
    """IDF-based feature interpretability using the trained SCAR model.

    Unlike analyze_sae_features() which uses the frozen SAE and ranks by
    total activation, this loads the full trained model (backbone LoRA +
    SAE LoRA + IDF weights) and ranks features by IDF weight — showing
    the rarest, most discriminative features the model actually uses.

    For each top-IDF feature, finds the 5 strongest-activating documents
    and extracts token spans with context.

    Output: Appendix E table for the paper.
    """
    import torch
    import json
    import numpy as np
    from collections import defaultdict, Counter
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    sae_vol.reload()
    retrieval_vol.reload()
    device = torch.device("cuda")
    log("[INTERP-IDF] Starting IDF-based feature interpretability analysis")

    # --- Load trained model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    # Load LoRA adapter
    lora_dir = f"{SAVE_DIR}/{run_name}/lora_adapter"
    retrieval_ckpt = f"{SAVE_DIR}/{run_name}/checkpoint_final.pt"
    if os.path.exists(lora_dir):
        model = PeftModel.from_pretrained(backbone, lora_dir)
        log(f"[INTERP-IDF] LoRA loaded from {lora_dir}")
    else:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(backbone, lora_config)
        ckpt = torch.load(retrieval_ckpt, map_location=device, weights_only=False)
        set_peft_model_state_dict(model, ckpt["model_state_dict"])
        log(f"[INTERP-IDF] LoRA loaded from checkpoint")
    model.eval()

    enable_bidirectional_attention(model)

    # --- Load SAE with LoRA ---
    sae_ckpt_path = f"{SAE_DIR}/{sae_run_name}/checkpoint_final.pt"
    sae, target_norm = load_frozen_sae(sae_ckpt_path, device, sae_lora_rank=sae_lora_rank)
    if sae_lora_rank > 0 and os.path.exists(retrieval_ckpt):
        ret_ckpt = torch.load(retrieval_ckpt, map_location=device, weights_only=False)
        if "sae_lora_state" in ret_ckpt:
            sae.load_state_dict(ret_ckpt["sae_lora_state"], strict=False)
            log(f"[INTERP-IDF] SAE LoRA loaded (rank={sae_lora_rank})")

    # --- Load IDF weights ---
    idf_weights = None
    if os.path.exists(retrieval_ckpt):
        ret_ckpt = torch.load(retrieval_ckpt, map_location=device, weights_only=False)
        if "idf_weights" in ret_ckpt:
            idf_weights = ret_ckpt["idf_weights"].to(device)
            log(f"[INTERP-IDF] IDF weights loaded ({idf_weights.shape})")

    if idf_weights is None:
        log("[INTERP-IDF] ERROR: no IDF weights found")
        return {"error": "no IDF weights"}

    log(f"[INTERP-IDF] IDF range: [{idf_weights.min().item():.2f}, {idf_weights.max().item():.2f}]")

    # --- Hook to capture Layer 19 activations ---
    _activations = {}
    target_layer = _get_target_layer(model, LAYER_IDX)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            _activations["hidden"] = output[0]
        else:
            _activations["hidden"] = output

    hook = target_layer.register_forward_hook(hook_fn)

    # --- Load corpus sample ---
    hf_token = os.environ.get("HF_TOKEN")
    from datasets import load_dataset
    log(f"[INTERP-IDF] Loading {n_corpus_samples} contracts from SAE corpus...")
    sae_ds = load_dataset(f"{HF_USERNAME}/scar-corpus", split="train", token=hf_token)
    sae_ds = sae_ds.shuffle(seed=42).select(range(min(n_corpus_samples, len(sae_ds))))
    corpus_texts = [row["contract_code"] for row in sae_ds]
    del sae_ds

    security_keywords = [
        'transfer', 'call', 'delegatecall', 'send', 'selfdestruct',
        'msg.sender', 'owner', 'admin', 'require', 'assert',
        'balance', 'withdraw', 'deposit', 'approve', 'allowance',
        'reentrancy', 'overflow', 'underflow', 'external', 'payable',
        'modifier', 'onlyOwner', 'auth', 'access', 'lock', 'guard',
        'slot', 'storage', 'proxy', 'implementation', 'upgrade',
        'oracle', 'price', 'flash', 'swap', 'liquidity', 'pool',
        'mint', 'burn', 'stake', 'reward', 'fee', 'slippage',
    ]

    # --- Stage 1: Scan corpus for ALL features, compute corpus doc frequency ---
    # Track per-feature: corpus_df, total_activation, and top-N doc examples
    min_corpus_df = 10  # feature must fire on at least 10 docs to be interpretable
    corpus_df = np.zeros(D_SAE, dtype=np.int32)  # doc frequency on corpus
    # For ALL features, store top-N docs (sparse — only store if feature fires)
    feature_docs = defaultdict(list)  # fid → [(doc_idx, sum_act, spans)]

    batch_size = 8
    log(f"[INTERP-IDF] Stage 1: Scanning {len(corpus_texts)} docs (all {D_SAE} features)...")
    for batch_start in range(0, len(corpus_texts), batch_size):
        batch_texts = corpus_texts[batch_start:batch_start + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_len,
        ).to(device)

        with torch.no_grad():
            model(**inputs)
            hidden = _activations["hidden"]

            # Normalize to SAE training scale
            norms = hidden.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            hidden_normed = hidden * (target_norm / norms)

            B, T, D = hidden_normed.shape

            for b in range(B):
                doc_idx = batch_start + b
                doc_hidden = hidden_normed[b:b+1]  # (1, T, D)
                mask = inputs["attention_mask"][b]  # (T,)
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][b])
                seq_len = int(mask.sum().item())

                flat = doc_hidden.reshape(-1, D)  # (T, D)
                _, z_pre = sae.encode(flat)  # (T, d_sae)
                z_pre = torch.relu(z_pre) * mask.float().unsqueeze(-1)  # (T, d_sae)

                # Per-feature sum activation for this doc
                doc_feat_sums = z_pre.sum(dim=0)  # (d_sae,)
                active_feats = (doc_feat_sums > 0).nonzero(as_tuple=True)[0]
                corpus_df[active_feats.cpu().numpy()] += 1

                # For active features, collect top-doc examples
                for fid_t in active_feats:
                    fid = fid_t.item()
                    sum_act = doc_feat_sums[fid].item()
                    docs = feature_docs[fid]

                    # Only store top-N docs per feature
                    if len(docs) < n_top_docs:
                        should_store = True
                    elif sum_act > min(d["sum_activation"] for d in docs):
                        should_store = True
                    else:
                        should_store = False

                    if should_store:
                        # Extract top-3 token spans for this feature
                        feat_acts = z_pre[:, fid]  # (T,)
                        n_active = int((feat_acts > 0).sum().item())
                        top_positions = feat_acts.topk(min(3, max(n_active, 1)))
                        spans = []
                        for k in range(top_positions.values.shape[0]):
                            t = top_positions.indices[k].item()
                            val = top_positions.values[k].item()
                            if val <= 0:
                                continue
                            tok_str = tokens[t] if t < len(tokens) else "?"
                            left_ctx = " ".join(tokens[max(0, t-5):t])
                            right_ctx = " ".join(tokens[t+1:min(len(tokens), t+6)])
                            spans.append({
                                "pos": t,
                                "activation": round(val, 3),
                                "token": tok_str,
                                "left": left_ctx,
                                "right": right_ctx,
                            })

                        doc_entry = {
                            "doc_idx": doc_idx,
                            "sum_activation": round(sum_act, 3),
                            "spans": spans,
                        }
                        if len(docs) < n_top_docs:
                            docs.append(doc_entry)
                        else:
                            min_idx = min(range(len(docs)),
                                          key=lambda i: docs[i]["sum_activation"])
                            docs[min_idx] = doc_entry

        if (batch_start + batch_size) % 500 == 0:
            log(f"  Processed {batch_start + batch_size}/{len(corpus_texts)} docs")

    hook.remove()

    # --- Stage 2: Select top-N features by IDF among those that fire ---
    active_feature_count = int((corpus_df >= min_corpus_df).sum())
    log(f"[INTERP-IDF] Stage 2: {active_feature_count} features with corpus_df>={min_corpus_df}")

    # Mask IDF: only keep features with sufficient corpus_df
    masked_idf = idf_weights.clone().cpu()
    for fid in range(D_SAE):
        if corpus_df[fid] < min_corpus_df:
            masked_idf[fid] = -1
    top_idf_vals, top_idf_ids = masked_idf.topk(n_top_features)
    log(f"[INTERP-IDF] Selected {n_top_features} features (IDF: "
        f"{top_idf_vals[-1].item():.4f}–{top_idf_vals[0].item():.4f}, "
        f"corpus_df: {corpus_df[top_idf_ids[-1].item()]}–{corpus_df[top_idf_ids[0].item()]})")

    # --- Build report ---
    log(f"\n[INTERP-IDF] ═══ TOP {n_top_features} SAE FEATURES (by IDF, corpus_df>={min_corpus_df}) ═══")
    feature_report = []

    for rank, fid in enumerate(top_idf_ids.tolist()):
        idf_val = idf_weights[fid].item()
        docs = sorted(feature_docs[fid], key=lambda d: -d["sum_activation"])

        # Collect all tokens across top docs
        all_tokens = []
        all_context = []
        for doc in docs:
            for span in doc["spans"]:
                all_tokens.append(span["token"])
                all_context.append(f"{span['left']} {span['token']} {span['right']}")
        token_counter = Counter(all_tokens)
        top_tokens = token_counter.most_common(5)

        # Check for security-relevant patterns
        context_str = " ".join(all_context).lower()
        sec_hits = [kw for kw in security_keywords if kw.lower() in context_str]

        # Auto-label: most common token pattern
        if top_tokens:
            label = top_tokens[0][0]
            if sec_hits:
                label = sec_hits[0]
        else:
            label = "inactive"

        feat_info = {
            "rank": rank + 1,
            "feature_id": fid,
            "idf": round(idf_val, 4),
            "corpus_df": int(corpus_df[fid]),
            "label": label,
            "top_tokens": top_tokens,
            "security_keywords": sec_hits[:5],
            "top_docs": docs,
        }
        feature_report.append(feat_info)

        # Print summary
        tok_summary = ", ".join(f"'{t}'({c})" for t, c in top_tokens[:3]) if top_tokens else "none"
        sec_label = f" SEC:[{','.join(sec_hits[:3])}]" if sec_hits else ""
        log(f"  #{rank+1} Feature {fid} (IDF={idf_val:.4f}, df={corpus_df[fid]}): "
            f"tokens=[{tok_summary}]{sec_label}")
        for doc in docs[:2]:
            for span in doc["spans"][:1]:
                log(f"      [{span['activation']:.2f}] "
                    f"...{span['left']} >>>{span['token']}<<< {span['right']}...")

    # --- Save report ---
    report = {
        "model": run_name,
        "n_top_features": n_top_features,
        "n_corpus_sampled": len(corpus_texts),
        "features": feature_report,
    }

    report_path = f"{SAVE_DIR}/interpretability_idf_report.json"
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    retrieval_vol.commit()

    log(f"\n[INTERP-IDF] Report saved to {report_path}")
    log("[INTERP-IDF] ANALYSIS COMPLETE")

    return report


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(mode: str = "primary", temperature: float = 0.5):
    """Run retrieval fine-tuning pipeline.

    Args:
        mode: "temperature" | "primary" | "all" | "improve_v2" | ...
        temperature: manual override (ignored if sweep results exist)
    """
    from datetime import datetime

    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"SCAR Retrieval Training (mode={mode})")

    best_tau = temperature

    if mode == "temperature":
        print("--- TEMPERATURE SWEEP ---")
        result = temperature_sweep.remote(
            candidates=[0.02, 0.05, 0.1, 0.5], force=True,
        )
        if result is not None:
            print(f"Best temperature: {result}")
        return

    if mode == "all":
        print("--- TEMPERATURE SWEEP ---")
        result = temperature_sweep.remote(
            candidates=[0.02, 0.05, 0.1, 0.5], force=True,
        )
        if result is not None:
            best_tau = result
            print(f"Using swept temperature={best_tau}")

    if mode in ("primary", "all"):
        print(f"\n--- PRIMARY TRAINING (tau={best_tau}) ---")
        train_retrieval.remote(
            temperature=best_tau,
            run_name="primary",
            force=True,
        )

    if mode == "seed_sweep":
        print(f"\n--- MULTI-RUN (tau={best_tau}, 5 non-deterministic trials) ---")
        for trial in range(5):
            print(f"\n--- Trial {trial} ---")
            train_retrieval.remote(
                temperature=best_tau,
                run_name=f"trial_{trial}",
                seed=None,  # non-deterministic, like the best run
                force=True,
            )

    if mode == "diagnose":
        print("\n--- DIAGNOSTIC: No LoRA + bidirectional + NO IDF ---")
        result1 = diagnose_retrieval.remote(use_lora=False, use_bidirectional=True, use_idf=False)
        print(f"Result: {result1}")
        print("\n--- DIAGNOSTIC: No LoRA + bidirectional + IDF ---")
        result2 = diagnose_retrieval.remote(use_lora=False, use_bidirectional=True, use_idf=True)
        print(f"Result: {result2}")
        print("\n--- DIAGNOSTIC: No LoRA + causal + IDF ---")
        result3 = diagnose_retrieval.remote(use_lora=False, use_bidirectional=False, use_idf=True)
        print(f"Result: {result3}")

    if mode == "improve":
        # Parallel experiments to improve standalone retriever performance.
        # Key levers: batch size (more negatives), corpus-level IDF, epochs.
        print("\n--- IMPROVEMENT SWEEP ---")

        # Experiment 1: corpus IDF (20k docs from SAE corpus)
        h1 = train_retrieval.spawn(
            temperature=best_tau, run_name="improve_corpus_idf",
            batch_size=16, micro_batch_size=16, epochs=2,
            idf_corpus_size=20000,
            force=True,
        )
        # Experiment 2: batch=32 + 3 epochs (more training with more negatives)
        h2 = train_retrieval.spawn(
            temperature=best_tau, run_name="improve_b32_3ep",
            batch_size=32, micro_batch_size=32, epochs=3,
            force=True,
        )
        # Experiment 3: corpus IDF + batch=32
        h3 = train_retrieval.spawn(
            temperature=best_tau, run_name="improve_b32_corpus_idf",
            batch_size=32, micro_batch_size=32, epochs=2,
            idf_corpus_size=20000,
            force=True,
        )

        for name, handle in [("corpus_idf", h1), ("b32_3ep", h2), ("b32_corpus_idf", h3)]:
            result = handle.get()
            print(f"  {name}: done")

    if mode == "improve_v2":
        # BM25 hard negative mining + synthetic pairs + larger batch + more epochs.
        # Key insight from SPLADE v2: BM25-mined negatives (ranks 5-50) provide
        # the single biggest retrieval improvement (+1.6 MRR).
        # Expected: 4,034 pairs → ~20-25k pairs via mining + synthetic generation.
        print(f"\n--- IMPROVE V2: BM25 Mining + Synthetic Pairs (tau={best_tau}) ---")
        train_retrieval.remote(
            temperature=best_tau,
            run_name="improve_v2",
            batch_size=32,
            micro_batch_size=32,
            epochs=5,
            use_bm25_mining=True,
            bm25_corpus_cap=50000,
            bm25_n_negatives=4,
            synthetic_max_pairs=8000,
            force=True,
        )

    if mode == "improve_v3":
        # Clean data + GradCache (effective batch=128) + temperature cool-down
        # + positive-aware BM25 hard negative filtering.
        # Uses the cleaned HF dataset (code-only positives, no query leakage).
        print(f"\n--- IMPROVE V3: GradCache + Temp Schedule + Clean Data ---")
        train_retrieval.remote(
            temperature=0.1,  # only used as initial if temp_schedule="fixed"
            run_name="improve_v3",
            batch_size=128,       # effective batch via GradCache
            micro_batch_size=16,  # fits H100 memory
            epochs=3,             # clean data → fewer epochs needed
            use_bm25_mining=True,
            bm25_corpus_cap=50000,
            bm25_n_negatives=4,
            synthetic_max_pairs=8000,
            use_gradcache=True,
            gradcache_chunk_size=16,
            temp_schedule="cosine_cooldown",
            temp_max=1.0,
            temp_min=0.05,
            force=True,
        )

    if mode == "improve_v4":
        # The full PRD vision: mixed-modality SAE + max-pool (SPLARE-style) +
        # GradCache + temp schedule + BM25 mining + clean data.
        # Uses the mixed SAE (mixed_L19_16k) trained on both code and queries.
        print(f"\n--- IMPROVE V4: Mixed SAE + Max-pool + GradCache ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="improve_v4",
            batch_size=128,
            micro_batch_size=16,
            epochs=5,
            use_bm25_mining=True,
            bm25_corpus_cap=50000,
            bm25_n_negatives=4,
            synthetic_max_pairs=8000,
            use_gradcache=True,
            gradcache_chunk_size=16,
            temp_schedule="cosine_cooldown",
            temp_max=1.0,
            temp_min=0.05,
            curriculum_easy_epochs=2,    # epochs 1-2: random negatives, 3-5: hard
            sae_run_name="mixed_L19_16k",  # use the modality-mixed SAE
            pooling_mode="max",  # SPLARE-style: preserves rare discriminative features
            force=True,
        )

    if mode == "v5a_simple":
        # Clean ablation: mixed SAE with primary's simple config
        # Isolates SAE effect without confounding BM25/curriculum/temp schedule
        print(f"\n--- V5A_SIMPLE: Mixed SAE + Primary config (clean ablation) ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v5a_simple",
            sae_run_name="mixed_L19_16k",  # only change from primary
            force=True,
        )

    if mode == "v5a":
        # Ablation: mixed SAE + sum-pool (isolate SAE effect from pooling)
        # v4 used max-pool and got 3% acc. Is it max-pool or the mixed SAE?
        print(f"\n--- V5A: Mixed SAE + Sum-pool (isolate pooling) ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v5a",
            batch_size=128,
            micro_batch_size=16,
            epochs=5,
            use_bm25_mining=True,
            bm25_corpus_cap=50000,
            bm25_n_negatives=4,
            synthetic_max_pairs=0,  # skip slow synthetic generation
            use_gradcache=True,
            gradcache_chunk_size=16,
            temp_schedule="cosine_cooldown",
            temp_max=1.0,
            temp_min=0.05,
            curriculum_easy_epochs=2,
            sae_run_name="mixed_L19_16k",  # same mixed SAE as v4
            pooling_mode="sum",  # KEY DIFFERENCE: sum-pool like primary/v3
            force=False,  # reuse cached BM25 pairs
        )

    if mode == "v5c":
        # SAE LoRA: adapt which SAE features fire for retrieval.
        # LoRA on W_enc with rank=16 (287k params). Uses primary SAE + simple config.
        # Attacks root cause: SAE features are generic → LoRA adapts them for discrimination.
        print(f"\n--- V5C: SAE LoRA (rank=16, primary SAE, simple config) ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v5c",
            sae_run_name="primary_L19_16k",  # primary SAE, NOT mixed
            sae_lora_rank=16,
            force=True,
        )

    if mode == "v5c_r32":
        # SAE LoRA rank=32: double capacity to adapt features. Tests if rank=16 is bottleneck.
        print(f"\n--- V5C_R32: SAE LoRA (rank=32, primary SAE, simple config) ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v5c_r32",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=32,
            force=True,
        )

    if mode == "v5c_r64_seeds":
        # 5 seeded runs for confidence intervals — parallel via .spawn()
        handles = []
        for s in [42, 123, 456, 789, 2026]:
            print(f"Spawning v5c_r64_s{s}...")
            h = train_retrieval.spawn(
                temperature=0.1,
                run_name=f"v5c_r64_s{s}",
                sae_run_name="primary_L19_16k",
                sae_lora_rank=64,
                seed=s,
                force=True,
            )
            handles.append((s, h))
        # Wait for all
        for s, h in handles:
            print(f"Waiting for v5c_r64_s{s}...")
            h.get()
            print(f"v5c_r64_s{s}: DONE")

    if mode == "v5c_r64_tau":
        # Temperature sweep: 0.05 and 0.2 (we already have 0.1) — parallel
        handles = []
        for tau in [0.05, 0.2]:
            name = f"v5c_r64_tau{str(tau).replace('.','_')}"
            print(f"Spawning {name}...")
            h = train_retrieval.spawn(
                temperature=tau,
                run_name=name,
                sae_run_name="primary_L19_16k",
                sae_lora_rank=64,
                force=True,
            )
            handles.append((name, h))
        for name, h in handles:
            print(f"Waiting for {name}...")
            h.get()
            print(f"{name}: DONE")

    if mode == "v5c_r64_lr2e4":
        # SAE LoRA rank=64 with 4x higher LR for SAE LoRA params
        print(f"\n--- V5C_R64_LR2E4: SAE LoRA rank=64, sae_lr=2e-4 ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v5c_r64_lr2e4",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=64,
            sae_lora_lr=2e-4,
            force=True,
        )

    if mode == "v5c_r64_lr5e4":
        # SAE LoRA rank=64 with 10x higher LR for SAE LoRA params
        print(f"\n--- V5C_R64_LR5E4: SAE LoRA rank=64, sae_lr=5e-4 ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v5c_r64_lr5e4",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=64,
            sae_lora_lr=5e-4,
            force=True,
        )

    if mode == "v5c_r128":
        print(f"\n--- V5C_R128: SAE LoRA (rank=128, primary SAE, simple config) ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v5c_r128",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=128,
            force=True,
        )

    if mode == "v5c_r64":
        # SAE LoRA rank=64: 4x v5c capacity. Tests scaling trend r16→r32→r64.
        print(f"\n--- V5C_R64: SAE LoRA (rank=64, primary SAE, simple config) ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v5c_r64",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=64,
            force=True,
        )

    if mode == "v5c_bb128":
        # Option 3: backbone LoRA rank 128 (was always 64). SAE-LoRA nearly saturated,
        # backbone may be the bottleneck for cross-modal query→code mapping.
        print(f"\n--- V5C_BB128: backbone LoRA rank=128, SAE LoRA rank=128 ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v5c_bb128",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=128,
            lora_rank=128,
            force=True,
        )

    if mode == "v5c_bb256":
        # Option 3 aggressive: backbone LoRA rank 256. Tests if backbone capacity
        # is the bottleneck. Risk: overfitting with 4k pairs.
        print(f"\n--- V5C_BB256: backbone LoRA rank=256, SAE LoRA rank=128 ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v5c_bb256",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=128,
            lora_rank=256,
            force=True,
        )

    if mode == "v5c_bb_sweep":
        # Run both backbone rank experiments in parallel
        print(f"\n--- V5C backbone rank sweep: r128 + r256 ---")
        h1 = train_retrieval.spawn(
            temperature=0.1,
            run_name="v5c_bb128",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=128,
            lora_rank=128,
            force=True,
        )
        h2 = train_retrieval.spawn(
            temperature=0.1,
            run_name="v5c_bb256",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=128,
            lora_rank=256,
            force=True,
        )
        print("Waiting for both runs...")
        h1.get()
        h2.get()
        print("Both backbone rank experiments complete.")

    if mode == "v5c_bm25":
        # SAE LoRA + BM25 hard negatives: tests if harder data helps when SAE can adapt.
        # improve_v3 (BM25 + frozen SAE) failed → but with SAE LoRA, harder negatives
        # give the SAE signal to learn discriminative features.
        print(f"\n--- V5C_BM25: SAE LoRA rank=16 + BM25 mining ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v5c_bm25",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=16,
            use_bm25_mining=True,
            bm25_corpus_cap=50000,
            bm25_n_negatives=4,
            synthetic_max_pairs=0,
            force=True,
        )

    if mode == "v6_data":
        # Retrain r64 + r128 on expanded dataset (7.5k pairs, up from 4k)
        # Parallel via .spawn()
        print(f"\n--- V6_DATA: r64 + r128 on expanded 7.5k dataset ---")
        h1 = train_retrieval.spawn(
            temperature=0.1,
            run_name="v6_r64",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=64,
            force=True,
        )
        h2 = train_retrieval.spawn(
            temperature=0.1,
            run_name="v6_r128",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=128,
            force=True,
        )
        print("Spawned both — waiting...")
        h1.get()
        print("v6_r64: DONE")
        h2.get()
        print("v6_r128: DONE")

    if mode == "v7_ce":
        # Step 1: Train cross-encoder for distillation
        print("\n--- V7: Train cross-encoder ---")
        result = train_cross_encoder.remote(force=True)
        print(f"Cross-encoder: {result}")

    if mode == "v7_scores":
        # Step 2: Pre-compute cross-encoder scores
        print("\n--- V7: Pre-compute distillation scores ---")
        result = precompute_ce_scores.remote()
        print(f"Scores: {result}")

    if mode == "v7_distill":
        # Step 3: Train with margin-MSE distillation (r64 + r128 in parallel)
        print(f"\n--- V7: Distillation training (r64 + r128) ---")
        h1 = train_retrieval.spawn(
            temperature=0.1,
            run_name="v7_r64",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=64,
            lambda_distill=0.5,
            force=True,
        )
        h2 = train_retrieval.spawn(
            temperature=0.1,
            run_name="v7_r128",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=128,
            lambda_distill=0.5,
            force=True,
        )
        print("Spawned both — waiting...")
        h1.get()
        print("v7_r64: DONE")
        h2.get()
        print("v7_r128: DONE")

    if mode == "v7_all":
        # Full pipeline: cross-encoder → scores → distill
        print("\n--- V7: Full distillation pipeline ---")
        print("Step 1: Training cross-encoder...")
        ce_result = train_cross_encoder.remote(force=True)
        print(f"Cross-encoder: val_acc={ce_result.get('val_acc', '?')}")
        if ce_result.get("val_acc", 0) < 0.70:
            print("WARNING: Cross-encoder accuracy < 70%. Distillation may hurt.")
        print("Step 2: Pre-computing scores...")
        sc_result = precompute_ce_scores.remote()
        print(f"Scores: margin_positive_pct={sc_result.get('margin_positive_pct', '?')}")
        print("Step 3: Distillation training (r64 + r128)...")
        h1 = train_retrieval.spawn(
            temperature=0.1, run_name="v7_r64",
            sae_run_name="primary_L19_16k", sae_lora_rank=64,
            lambda_distill=0.5, force=True,
        )
        h2 = train_retrieval.spawn(
            temperature=0.1, run_name="v7_r128",
            sae_run_name="primary_L19_16k", sae_lora_rank=128,
            lambda_distill=0.5, force=True,
        )
        print("Spawned both — waiting...")
        h1.get()
        print("v7_r64: DONE")
        h2.get()
        print("v7_r128: DONE")

    if mode == "v8_seq512":
        # v8: Retrain best config (v7_r128 distillation) with max_seq_len=512 instead of 256
        print(f"\n--- V8: seq_len=512, distillation, r64 + r128 ---")
        h1 = train_retrieval.spawn(
            temperature=0.1, run_name="v8_r64",
            sae_run_name="primary_L19_16k", sae_lora_rank=64,
            lambda_distill=0.5, max_seq_len=512, micro_batch_size=16, force=True,
        )
        h2 = train_retrieval.spawn(
            temperature=0.1, run_name="v8_r128",
            sae_run_name="primary_L19_16k", sae_lora_rank=128,
            lambda_distill=0.5, max_seq_len=512, micro_batch_size=16, force=True,
        )
        print("Spawned both — waiting...")
        h1.get()
        print("v8_r64: DONE")
        h2.get()
        print("v8_r128: DONE")

    if mode == "v9":
        # v9: Retrain best config (v7_r128) on expanded scar-pairs (17,578 pairs)
        # Full pipeline: cross-encoder → distillation scores → retrieval training
        print("\n--- V9: Retrain on scar-pairs (17,578 pairs) ---")
        print("Step 1: Training cross-encoder on scar-pairs...")
        ce_result = train_cross_encoder.remote(force=True)
        print(f"Cross-encoder: val_acc={ce_result.get('val_acc', '?')}")
        print("Step 2: Pre-computing distillation scores...")
        sc_result = precompute_ce_scores.remote()
        print(f"Scores: margin_positive_pct={sc_result.get('margin_positive_pct', '?')}")
        print("Step 3: Retrieval training (v7_r128 config + scar-pairs)...")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v9",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=128,
            lambda_distill=0.5,
            force=True,
        )
        print("v9: DONE")

    if mode == "v10":
        # v10: Retrain on quality-filtered scar-pairs (11,961 pairs)
        print("\n--- V10: Retrain on quality-filtered scar-pairs (11,961 pairs) ---")
        print("Step 1: Training cross-encoder on filtered scar-pairs...")
        ce_result = train_cross_encoder.remote(force=True)
        print(f"Cross-encoder: val_acc={ce_result.get('val_acc', '?')}")
        print("Step 2: Pre-computing distillation scores...")
        sc_result = precompute_ce_scores.remote()
        print(f"Scores: margin_positive_pct={sc_result.get('margin_positive_pct', '?')}")
        print("Step 3: Retrieval training (v7_r128 config + filtered data)...")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v10",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=128,
            lambda_distill=0.5,
            force=True,
        )
        print("v10: DONE")

    if mode == "v11":
        # v11: Enhance v7 — two parallel experiments on v7's original data
        # v7 used scar-pairs (old dataset, ~6.8k pairs)
        OLD_PAIRS = f"{HF_USERNAME}/scar-pairs"
        print("\n--- V11: Enhance v7 (rank=256 + epochs=10) ---")
        print("Using v7's original cross-encoder + distillation scores")
        # Launch both in parallel via spawn
        print("Launching v11_r256 (rank=256, 5 epochs)...")
        train_retrieval.spawn(
            temperature=0.1,
            run_name="v11_r256",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=256,
            lambda_distill=0.5,
            pairs_dataset=OLD_PAIRS,
            force=True,
        )
        print("Launching v11_e10 (rank=128, 10 epochs)...")
        train_retrieval.spawn(
            temperature=0.1,
            run_name="v11_e10",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=128,
            epochs=10,
            lambda_distill=0.5,
            pairs_dataset=OLD_PAIRS,
            force=True,
        )
        print("v11: Both launched (check Modal dashboard)")

    if mode == "v11_combo":
        # v11_combo: rank=256 + 10 epochs — combine both winners
        OLD_PAIRS = f"{HF_USERNAME}/scar-pairs"
        print("\n--- V11_COMBO: rank=256 + 10 epochs ---")
        train_retrieval.remote(
            temperature=0.1,
            run_name="v11_combo",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=256,
            epochs=10,
            lambda_distill=0.5,
            pairs_dataset=OLD_PAIRS,
            force=True,
        )
        print("v11_combo: DONE")

    if mode == "v13":
        # v13: Push epochs further — 20 and 25
        OLD_PAIRS = f"{HF_USERNAME}/scar-pairs"
        print("\n--- V13: More epochs (20 + 25) ---")
        print("Launching v13_e20 (rank=256, 20 epochs)...")
        train_retrieval.spawn(
            temperature=0.1,
            run_name="v13_e20",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=256,
            epochs=20,
            lambda_distill=0.5,
            pairs_dataset=OLD_PAIRS,
            force=True,
        )
        print("Launching v13_e25 (rank=256, 25 epochs)...")
        train_retrieval.spawn(
            temperature=0.1,
            run_name="v13_e25",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=256,
            epochs=25,
            lambda_distill=0.5,
            pairs_dataset=OLD_PAIRS,
            force=True,
        )
        print("v13: Both launched")

    if mode == "v14":
        # v14: Ablation — no distillation (same as v13_e25 but lambda_distill=0)
        OLD_PAIRS = f"{HF_USERNAME}/scar-pairs"
        print("\n--- V14: Distillation ablation (no distill, 25 epochs, r256) ---")
        train_retrieval.spawn(
            temperature=0.1,
            run_name="v14_nodistill",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=256,
            epochs=25,
            lambda_distill=0.0,
            pairs_dataset=OLD_PAIRS,
            force=True,
        )
        print("v14_nodistill launched")

    if mode == "v15":
        # v15: Deconfound ablation — rank 128, 5 epochs, NO distillation
        # Fills the missing cell in Table 2: what does rank 128 alone do (without distill)?
        OLD_PAIRS = f"{HF_USERNAME}/scar-pairs"
        print("\n--- V15: Deconfound ablation (r128, 5ep, no distill) ---")
        train_retrieval.spawn(
            temperature=0.1,
            run_name="v15_r128_nodistill",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=128,
            epochs=5,
            lambda_distill=0.0,
            pairs_dataset=OLD_PAIRS,
            force=True,
        )
        print("v15_r128_nodistill launched")

    if mode == "v16":
        # v16: SAE-LoRA-only ablation — freeze backbone entirely (no backbone LoRA),
        # train only SAE-LoRA rank 256. Isolates SAE-LoRA contribution.
        OLD_PAIRS = f"{HF_USERNAME}/scar-pairs"
        print("\n--- V16: SAE-LoRA-only ablation (no backbone LoRA, r256, 25ep) ---")
        train_retrieval.spawn(
            temperature=0.1,
            run_name="v16_sae_only",
            sae_run_name="primary_L19_16k",
            lora_rank=0,  # NO backbone LoRA
            sae_lora_rank=256,
            epochs=25,
            lambda_distill=0.0,
            pairs_dataset=OLD_PAIRS,
            force=True,
        )
        print("v16_sae_only launched")

    if mode == "v12":
        # v12: All remaining experiments in parallel
        OLD_PAIRS = f"{HF_USERNAME}/scar-pairs"
        print("\n--- V12: Parallel experiments ---")

        # v12_e15: 15 epochs, rank=256 (push further)
        print("Launching v12_e15 (rank=256, 15 epochs)...")
        train_retrieval.spawn(
            temperature=0.1,
            run_name="v12_e15",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=256,
            epochs=15,
            lambda_distill=0.5,
            pairs_dataset=OLD_PAIRS,
            force=True,
        )

        # v12_bb128: backbone LoRA rank 128 (double backbone capacity)
        print("Launching v12_bb128 (backbone r=128, SAE r=256, 10 epochs)...")
        train_retrieval.spawn(
            temperature=0.1,
            run_name="v12_bb128",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=256,
            lora_rank=128,
            epochs=10,
            lambda_distill=0.5,
            pairs_dataset=OLD_PAIRS,
            force=True,
        )

        # v12_lr3e5: lower LR for better convergence
        print("Launching v12_lr3e5 (lr=3e-5, rank=256, 10 epochs)...")
        train_retrieval.spawn(
            temperature=0.1,
            run_name="v12_lr3e5",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=256,
            lr=3e-5,
            epochs=10,
            lambda_distill=0.5,
            pairs_dataset=OLD_PAIRS,
            force=True,
        )

        # v12_data: filtered 11.9k data + 10 epochs (data + epochs combo)
        print("Launching v12_data (filtered scar-pairs, rank=256, 10 epochs)...")
        train_retrieval.spawn(
            temperature=0.1,
            run_name="v12_data",
            sae_run_name="primary_L19_16k",
            sae_lora_rank=256,
            epochs=10,
            lambda_distill=0.5,
            force=True,  # uses default PAIRS_DATASET (scar-pairs, 11.9k)
        )

        print("v12: All 4 launched (check Modal dashboard)")

    if mode == "splade":
        # SPLADE baseline: same backbone + LoRA, but LM head (vocab-space, 152k dims)
        # instead of SAE encoder (16k dims). Isolates the SAE contribution.
        # micro_batch=8 + grad_accum=4 → effective batch=32 (logits OOM at batch=32)
        OLD_PAIRS = f"{HF_USERNAME}/scar-pairs"
        print("\n--- SPLADE BASELINE (vocab-space, no SAE) ---")
        train_retrieval.remote(
            run_name="splade_v1",
            retriever_type="splade",
            temperature=0.1,
            lr=5e-5,
            batch_size=32,
            micro_batch_size=8,
            epochs=25,
            lora_rank=64,
            lambda_d=1e-6,
            lambda_q=1e-6,
            topk_query=100,
            topk_doc=400,
            sae_lora_rank=0,
            lambda_distill=0.0,
            pairs_dataset=OLD_PAIRS,
            force=True,
        )

    if mode == "interpret":
        # SAE feature interpretability analysis — THE critical diagnostic.
        # Answers: do SAE features capture security-relevant code patterns?
        print("\n--- SAE FEATURE INTERPRETABILITY ANALYSIS ---")
        report = analyze_sae_features.remote(
            n_corpus_samples=2000,
            n_top_features=50,
            n_top_tokens_per_feature=20,
        )
        print(f"\nActive features: {report['total_active_features']}/{report['total_sae_features']}")
        print(f"Feature utilization: {report['feature_utilization']*100:.1f}%")
        mg = report['modality_gap']
        print(f"Modality gap — Query-only: {mg['query_only_features']}, "
              f"Doc-only: {mg['doc_only_features']}, "
              f"Shared: {mg['shared_features']}, "
              f"Jaccard: {mg['jaccard']:.3f}")

    if mode == "interpret_idf":
        # IDF-based feature interpretability using trained v13_e25 model.
        # Top-10 highest-IDF features with strongest-activating documents.
        print("\n--- IDF-BASED FEATURE INTERPRETABILITY (trained model) ---")
        report = analyze_sae_features_idf.remote(
            run_name="v13_e25",
            sae_lora_rank=256,
            n_corpus_samples=2000,
            n_top_features=10,
            n_top_docs=5,
        )
        if report and "error" not in report:
            print(f"\nTop {report['n_top_features']} features by IDF:")
            for feat in report["features"]:
                sec = f" [{','.join(feat['security_keywords'][:3])}]" if feat['security_keywords'] else ""
                print(f"  #{feat['rank']} Feature {feat['feature_id']} "
                      f"(IDF={feat['idf']:.4f}): {feat['label']}{sec}")
        else:
            print(f"Error: {report}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
