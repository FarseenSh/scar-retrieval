"""
SCAR Step 5: SAE Pretraining — Modal Labs
=====================================================
Trains a JumpReLU Sparse Autoencoder on Qwen2.5-Coder-1.5B Layer 19
residual stream activations using 231k Solidity contracts.

Follows the JumpReLU SAE training recipe (Rajamanoharan et al., 2024):
  - JumpReLU with learned per-feature thresholds
  - Tanh sparsity loss + pre-activation dead feature loss
  - Activation normalization (E[||x||] = sqrt(d_in))
  - Decoder row normalization after each step
  - Linear lambda_S warmup over entire training

Usage:
  modal run scripts/step5_sae.py                          # primary (L19, 16k)
  modal run scripts/step5_sae.py --mode calibrate         # find lambda_s
  modal run scripts/step5_sae.py --mode ablations         # all ablations
  modal run scripts/step5_sae.py --mode all               # everything
  modal run scripts/step5_sae.py --mode inspect            # feature inspection
  modal run scripts/step5_sae.py --mode mixed              # mixed-modality SAE
"""

import modal
import os
import math

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("scar-sae")

vol = modal.Volume.from_name("scar-sae-training", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0,<2.6.0",
        "transformers>=4.45.0,<5.0.0",
        "datasets>=2.19.0,<4.0.0",
        "huggingface_hub>=0.23.0",
        "safetensors>=0.4.0",
        "tqdm>=4.66.0",
    )
)

SAVE_DIR = "/sae_training"
HF_SECRET = modal.Secret.from_name("huggingface-token")
HF_USERNAME = "Farseen0"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
DATASET_NAME = f"{HF_USERNAME}/scar-corpus"
D_IN = 1536  # Qwen2.5-Coder-1.5B hidden_size


def log(msg: str):
    """Timestamped logging with forced flush for Modal containers."""
    import sys
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: JumpReLU SAE Architecture
# ═══════════════════════════════════════════════════════════════════════════

def build_sae_class():
    """Returns the JumpReLUSAE class. Called inside Modal functions to avoid
    top-level torch import."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class JumpReLUSAE(nn.Module):
        """JumpReLU Sparse Autoencoder following Rajamanoharan et al. (2024).

        Architecture:
            Encoder: x_centered @ W_enc + b_enc -> z_pre
            JumpReLU: max(0, z_pre - threshold) -> z
            Decoder: z @ W_dec + b_dec -> x_hat

        Loss:
            L = MSE(x, x_hat) + lambda_s * L_sparsity + L_preact
            L_sparsity = mean(sum(tanh(c * |z_pre| / ||W_dec_row||)))
            L_preact = lambda_p * mean(sum(ReLU(theta - z_pre) * ||W_dec_row||))
        """

        def __init__(self, d_in: int = D_IN, d_sae: int = 16384,
                     init_threshold: float = 0.1):
            super().__init__()
            self.d_in = d_in
            self.d_sae = d_sae

            # Encoder
            self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
            self.b_enc = nn.Parameter(torch.zeros(d_sae))

            # Decoder
            self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
            self.b_dec = nn.Parameter(torch.zeros(d_in))

            # Learned per-feature threshold (stored as log for numerical stability)
            self.log_threshold = nn.Parameter(
                torch.full((d_sae,), math.log(init_threshold))
            )

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            # Kaiming-style init scaled by input dim
            nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
            # Normalize decoder rows to unit norm
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

        @property
        def threshold(self):
            return self.log_threshold.exp()

        def encode(self, x):
            """Returns (z, z_pre) where z is post-JumpReLU activations."""
            x_centered = x - self.b_dec
            z_pre = x_centered @ self.W_enc + self.b_enc
            z = F.relu(z_pre - self.threshold)
            return z, z_pre

        def decode(self, z):
            return z @ self.W_dec + self.b_dec

        def forward(self, x):
            z, z_pre = self.encode(x)
            x_hat = self.decode(z)
            return x_hat, z, z_pre

        def compute_loss(self, x, x_hat, z, z_pre, lambda_s, lambda_p=3e-6,
                         c=4.0):
            """Compute total loss: MSE + tanh sparsity + pre-act.

            Args:
                x: input activations (batch, d_in)
                x_hat: reconstructed activations (batch, d_in)
                z: post-JumpReLU activations (batch, d_sae)
                z_pre: pre-JumpReLU activations (batch, d_sae)
                lambda_s: sparsity loss coefficient (warmed up during training)
                lambda_p: pre-activation loss coefficient (default 3e-6)
                c: tanh scaling factor (default 4.0)

            Returns:
                (total_loss, metrics_dict)
            """
            # Reconstruction loss
            mse_loss = F.mse_loss(x_hat, x)

            # Decoder row norms for scaling (each row = one feature's direction)
            W_dec_norms = self.W_dec.norm(dim=1).clamp(min=1e-8)  # (d_sae,)

            # Tanh sparsity loss (Rajamanoharan et al., 2024)
            scaled = c * z_pre.abs() / W_dec_norms.unsqueeze(0)
            sparsity_loss = scaled.tanh().sum(dim=-1).mean()

            # Pre-activation loss (prevents dead features)
            pre_act = F.relu(self.threshold.unsqueeze(0) - z_pre)
            pre_act_loss = (pre_act * W_dec_norms.unsqueeze(0)).sum(dim=-1).mean()

            total = mse_loss + lambda_s * sparsity_loss + lambda_p * pre_act_loss

            # Metrics
            l0 = (z > 0).float().sum(dim=-1).mean().item()
            var_x = ((x - x.mean(dim=0, keepdim=True)) ** 2).sum()
            var_resid = ((x - x_hat) ** 2).sum()
            fvu = (var_resid / var_x.clamp(min=1e-8)).item()

            metrics = {
                "mse": mse_loss.item(),
                "sparsity": sparsity_loss.item(),
                "pre_act": pre_act_loss.item(),
                "total": total.item(),
                "l0": l0,
                "fvu": fvu,
                "var_explained": 1.0 - fvu,
                "mean_threshold": self.threshold.mean().item(),
            }
            return total, metrics

        def normalize_decoder(self):
            """Normalize decoder rows to unit norm (call after optimizer.step())."""
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    return JumpReLUSAE


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Helper Classes
# ═══════════════════════════════════════════════════════════════════════════

class ActivationNormalizer:
    """Maintains running mean of activation norms and normalizes to target.

    Target: E[||x||_2] = sqrt(d_in) per the JumpReLU SAE recipe.
    """

    def __init__(self, d_in: int, momentum: float = 0.01):
        self.target_norm = math.sqrt(d_in)
        self.running_mean_norm = None
        self.momentum = momentum

    def normalize(self, x):
        """Normalize activation batch. x: (batch, d_in) tensor."""
        import torch
        batch_mean_norm = x.norm(dim=-1).mean().item()
        if self.running_mean_norm is None:
            self.running_mean_norm = batch_mean_norm
        else:
            self.running_mean_norm = (
                (1 - self.momentum) * self.running_mean_norm
                + self.momentum * batch_mean_norm
            )
        if self.running_mean_norm > 1e-8:
            scale = self.target_norm / self.running_mean_norm
            return x * scale
        return x

    def state_dict(self):
        return {
            "running_mean_norm": self.running_mean_norm,
            "target_norm": self.target_norm,
        }

    def load_state_dict(self, d):
        self.running_mean_norm = d["running_mean_norm"]
        self.target_norm = d["target_norm"]


class DeadFeatureTracker:
    """Tracks which SAE features are 'dead' (haven't fired recently)."""

    def __init__(self, d_sae: int, window: int = 10000):
        self.window = window
        self.d_sae = d_sae
        self.last_fired = None  # lazy init (needs torch)

    def update(self, z, step: int):
        """Update with post-JumpReLU activations z: (batch, d_sae)."""
        import torch
        if self.last_fired is None:
            self.last_fired = torch.zeros(self.d_sae, dtype=torch.long,
                                          device=z.device)
        fired = (z > 0).any(dim=0)
        self.last_fired[fired] = step

    def dead_fraction(self, step: int) -> float:
        if self.last_fired is None:
            return 0.0
        dead = (step - self.last_fired) > self.window
        return dead.float().mean().item()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Data Streaming + Activation Collection
# ═══════════════════════════════════════════════════════════════════════════

def stream_activations(model, tokenizer, layer_idx, max_seq_len, model_batch_size,
                       device, hf_token=None, max_contracts=None):
    """Generator that streams residual stream activations from the model.

    Loads contracts from HuggingFace, packs into fixed-length sequences,
    runs through frozen model, captures target layer via forward hook.

    Yields:
        torch.Tensor of shape (n_tokens, d_in) in fp32
    """
    import torch
    from datasets import load_dataset

    # Load dataset
    ds = load_dataset(DATASET_NAME, split="train", token=hf_token)
    if max_contracts is not None:
        ds = ds.select(range(min(max_contracts, len(ds))))

    log(f"Streaming activations from {len(ds)} contracts, layer {layer_idx}, "
        f"seq_len {max_seq_len}, batch {model_batch_size}")

    # Token packing buffer
    token_buffer = []
    packed_sequences = []

    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    n_contracts_processed = 0

    total_batches_yielded = 0
    n_forward_passes = 0
    import time as _time

    for i, row in enumerate(ds):
        n_contracts_processed = i + 1
        if n_contracts_processed <= 3:
            log(f"  contract #{n_contracts_processed}: has 'contract_code'={('contract_code' in row)}, "
                f"keys={list(row.keys())[:5]}")
        if n_contracts_processed % 500 == 0:
            log(f"  streaming progress: {n_contracts_processed}/{len(ds)} contracts, "
                f"{total_batches_yielded} batches yielded, {n_forward_passes} fwd passes, "
                f"buf={len(token_buffer)} toks, packed={len(packed_sequences)} seqs")
        code = row.get("contract_code", "")
        if not code or not code.strip():
            if n_contracts_processed <= 5:
                log(f"  contract #{n_contracts_processed}: EMPTY, skipping")
            continue
        tokens = tokenizer.encode(code, add_special_tokens=False)
        if n_contracts_processed <= 3:
            log(f"  contract #{n_contracts_processed}: encoded to {len(tokens)} tokens")
        token_buffer.extend(tokens)
        token_buffer.append(eos_id)

        # Slice off complete sequences
        while len(token_buffer) >= max_seq_len:
            chunk = token_buffer[:max_seq_len]
            token_buffer = token_buffer[max_seq_len:]
            packed_sequences.append(chunk)

            # When we have a full model batch, run forward pass
            if len(packed_sequences) >= model_batch_size:
                batch_tokens = torch.tensor(packed_sequences, dtype=torch.long,
                                            device=device)
                packed_sequences = []

                if n_forward_passes < 3:
                    log(f"  fwd pass #{n_forward_passes}: input shape={batch_tokens.shape}, "
                        f"dtype={batch_tokens.dtype}, max_id={batch_tokens.max().item()}")

                # Forward pass with hook
                activations_captured = []

                def hook_fn(module, input, output):
                    # Handle both tuple (older transformers) and raw tensor (newer)
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    activations_captured.append(hidden.detach().float())

                handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
                t0 = _time.time()
                with torch.no_grad():
                    model(input_ids=batch_tokens, use_cache=False)
                dt = _time.time() - t0
                handle.remove()
                n_forward_passes += 1

                if n_forward_passes <= 3:
                    cap_shape = activations_captured[0].shape if activations_captured else "empty"
                    log(f"  fwd pass #{n_forward_passes}: done in {dt:.2f}s, "
                        f"captured={len(activations_captured)} tensors, "
                        f"captured_shape={cap_shape}")

                if activations_captured:
                    # Flatten from (batch, seq_len, d_in) to (batch*seq_len, d_in)
                    acts = activations_captured[0].reshape(-1, D_IN)
                    total_batches_yielded += 1
                    if total_batches_yielded <= 3:
                        log(f"  yielding batch #{total_batches_yielded}: shape={acts.shape}")
                    yield acts

                # Free memory
                del activations_captured, batch_tokens
                if n_forward_passes % 100 == 0:
                    torch.cuda.empty_cache()

    # Handle remaining sequences (partial batch)
    if packed_sequences:
        batch_tokens = torch.tensor(packed_sequences, dtype=torch.long,
                                    device=device)
        activations_captured = []

        def hook_fn_final(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            activations_captured.append(hidden.detach().float())

        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn_final)
        with torch.no_grad():
            model(input_ids=batch_tokens, use_cache=False)
        handle.remove()

        if activations_captured:
            acts = activations_captured[0].reshape(-1, D_IN)
            yield acts

        del activations_captured, batch_tokens
        torch.cuda.empty_cache()

    log(f"Activation streaming complete (processed {n_contracts_processed} contracts)")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4b: Mixed-Modality Activation Streaming
# ═══════════════════════════════════════════════════════════════════════════

PAIRS_DATASET = f"{HF_USERNAME}/scar-pairs"

def stream_mixed_activations(model, tokenizer, layer_idx, max_seq_len, model_batch_size,
                             device, hf_token=None, max_contracts=None,
                             code_to_query_ratio=4):
    """Stream activations from BOTH code (SAE corpus) AND queries (contrastive pairs).

    Interleaves code and query activations so the SAE learns features shared across
    both modalities. Without this, the SAE only sees code → features are code-specific
    → queries activate only generic features → retrieval fails.

    Interleaving ratio: for every `code_to_query_ratio` code batches, yield 1 query batch.
    Default 4:1 maintains code as the majority while exposing the SAE to vulnerability
    descriptions.

    Yields:
        torch.Tensor of shape (n_tokens, d_in) in fp32
    """
    import torch
    from datasets import load_dataset

    # Load contrastive pairs for query activations
    pairs_ds = load_dataset(PAIRS_DATASET, split="train", token=hf_token)
    query_texts = [row["query"] for row in pairs_ds]
    positive_texts = [row["positive"] for row in pairs_ds]
    # Mix queries and positives — both are text the SAE needs to understand
    mixed_texts = []
    for q, p in zip(query_texts, positive_texts):
        mixed_texts.append(q)
        mixed_texts.append(p)
    log(f"Mixed streaming: {len(mixed_texts)} query+positive texts from {len(pairs_ds)} pairs")
    del pairs_ds, query_texts, positive_texts

    # Tokenize all query/positive texts into packed sequences
    query_token_buffer = []
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    for text in mixed_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        query_token_buffer.extend(tokens)
        query_token_buffer.append(eos_id)
    del mixed_texts

    # Slice into fixed-length sequences
    query_sequences = []
    while len(query_token_buffer) >= max_seq_len:
        query_sequences.append(query_token_buffer[:max_seq_len])
        query_token_buffer = query_token_buffer[max_seq_len:]
    log(f"Mixed streaming: {len(query_sequences)} query sequences (len={max_seq_len})")

    # Query batch generator
    query_batch_idx = 0

    def get_query_batch():
        nonlocal query_batch_idx
        if query_batch_idx + model_batch_size > len(query_sequences):
            query_batch_idx = 0  # wrap around
        batch_seqs = query_sequences[query_batch_idx:query_batch_idx + model_batch_size]
        query_batch_idx += model_batch_size
        if not batch_seqs:
            return None

        batch_tokens = torch.tensor(batch_seqs, dtype=torch.long, device=device)

        activations_captured = []
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            activations_captured.append(hidden.detach().float())

        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(input_ids=batch_tokens, use_cache=False)
        handle.remove()

        if activations_captured:
            acts = activations_captured[0].reshape(-1, D_IN)
            return acts
        return None

    # Interleave code and query activations
    code_batch_count = 0
    total_code_batches = 0
    total_query_batches = 0

    for code_acts in stream_activations(
        model, tokenizer, layer_idx, max_seq_len, model_batch_size, device,
        hf_token=hf_token, max_contracts=max_contracts,
    ):
        yield code_acts
        total_code_batches += 1
        code_batch_count += 1

        # Every N code batches, inject a query batch
        if code_batch_count >= code_to_query_ratio:
            query_acts = get_query_batch()
            if query_acts is not None:
                yield query_acts
                total_query_batches += 1
            code_batch_count = 0

        if (total_code_batches + total_query_batches) % 500 == 0:
            log(f"  mixed streaming: {total_code_batches} code + "
                f"{total_query_batches} query batches yielded")

    log(f"Mixed streaming complete: {total_code_batches} code + "
        f"{total_query_batches} query batches "
        f"(ratio {total_code_batches / max(total_query_batches, 1):.1f}:1)")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: SAE Training Function
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={SAVE_DIR: vol},
    secrets=[HF_SECRET],
    timeout=86400,  # 24 hours max
)
def train_sae(
    layer: int = 19,
    d_sae: int = 16384,
    run_name: str = "primary_L19_16k",
    lambda_s_final: float = 0.01,
    lambda_p: float = 3e-6,
    lr: float = 2e-4,
    sae_batch_size: int = 4096,
    model_batch_size: int = 8,
    max_seq_len: int = 1024,
    max_contracts: int = None,
    c: float = 4.0,
    grad_clip: float = 1.0,
    init_threshold: float = 0.1,
    log_every: int = 100,
    checkpoint_pcts: list = None,
):
    """Train JumpReLU SAE on residual stream activations.

    Online streaming: loads model + SAE, streams dataset, trains SAE in one pass.
    """
    import torch
    import torch.nn.functional as F
    import json
    from tqdm import tqdm

    if checkpoint_pcts is None:
        checkpoint_pcts = [25, 50, 75, 100]

    vol.reload()

    run_dir = f"{SAVE_DIR}/{run_name}"
    final_ckpt = f"{run_dir}/checkpoint_final.pt"

    # Skip if already done
    if os.path.exists(final_ckpt):
        log(f"{run_name}: already trained, skipping")
        return

    # Auto-load calibrated lambda from volume if available
    cal_path = f"{SAVE_DIR}/calibration/lambda_s_sweep.json"
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            cal_data = json.load(f)
        if "best_lambda_s" in cal_data:
            calibrated = cal_data["best_lambda_s"]
            log(f"{run_name}: using calibrated lambda_s={calibrated} "
                f"(L0={cal_data[str(calibrated)].get('avg_l0', '?')})")
            lambda_s_final = calibrated

    os.makedirs(run_dir, exist_ok=True)

    # Save config
    config = {
        "run_name": run_name, "layer": layer, "d_in": D_IN, "d_sae": d_sae,
        "lambda_s_final": lambda_s_final, "lambda_p": lambda_p, "lr": lr,
        "sae_batch_size": sae_batch_size, "model_batch_size": model_batch_size,
        "max_seq_len": max_seq_len, "c": c, "grad_clip": grad_clip,
        "init_threshold": init_threshold, "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
    }
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{run_name}: device={device}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    # Load model (frozen, fp16)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"{run_name}: loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(device).eval()

    for param in model.parameters():
        param.requires_grad = False

    log(f"{run_name}: model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    # Initialize SAE
    JumpReLUSAE = build_sae_class()
    sae = JumpReLUSAE(d_in=D_IN, d_sae=d_sae, init_threshold=init_threshold).to(device)
    sae_params = sum(p.numel() for p in sae.parameters())
    log(f"{run_name}: SAE initialized ({sae_params / 1e6:.1f}M params, {d_sae} features)")

    # Check for resume checkpoint
    resume_step = 0
    resume_ckpt = None
    for pct in sorted(checkpoint_pcts[:-1], reverse=True):
        ckpt_path = f"{run_dir}/checkpoint_{pct}pct.pt"
        if os.path.exists(ckpt_path):
            resume_ckpt = ckpt_path
            break

    # Optimizer
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    normalizer = ActivationNormalizer(D_IN)
    dead_tracker = DeadFeatureTracker(d_sae)

    if resume_ckpt:
        log(f"{run_name}: resuming from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        sae.load_state_dict(ckpt["sae_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        normalizer.load_state_dict(ckpt["normalizer_state_dict"])
        resume_step = ckpt["step"]
        log(f"{run_name}: resumed at step {resume_step}")

    # Estimate total steps
    # ~231k contracts × ~1500 avg tokens / (sae_batch_size) ≈ 84k steps
    # Rough estimate; actual count determined by streaming
    est_total_tokens = 231000 * 1500
    est_total_steps = est_total_tokens // sae_batch_size
    log(f"{run_name}: estimated ~{est_total_steps} steps (~{est_total_tokens / 1e6:.0f}M tokens)")

    # Checkpoint step targets
    ckpt_steps = {int(est_total_steps * pct / 100): pct for pct in checkpoint_pcts}

    hf_token = os.environ.get("HF_TOKEN")

    # --- Training loop ---
    step = 0
    act_buffer = []
    act_buffer_size = 0
    training_log = []

    log(f"{run_name}: starting training (layer={layer}, d_sae={d_sae}, "
        f"lambda_s_final={lambda_s_final})")

    for act_batch in stream_activations(
        model, tokenizer, layer, max_seq_len, model_batch_size, device,
        hf_token=hf_token, max_contracts=max_contracts,
    ):
        # Accumulate in buffer
        act_buffer.append(act_batch)
        act_buffer_size += act_batch.shape[0]

        # Train when we have enough tokens
        while act_buffer_size >= sae_batch_size:
            # Assemble SAE batch
            all_acts = torch.cat(act_buffer, dim=0)
            sae_input = all_acts[:sae_batch_size]
            remainder = all_acts[sae_batch_size:]
            act_buffer = [remainder] if remainder.shape[0] > 0 else []
            act_buffer_size = remainder.shape[0] if remainder.shape[0] > 0 else 0

            # Shuffle
            perm = torch.randperm(sae_input.shape[0], device=device)
            sae_input = sae_input[perm]

            # Skip steps if resuming
            if step < resume_step:
                step += 1
                continue

            # Normalize activations
            sae_input = normalizer.normalize(sae_input)

            # Lambda_S warmup (linear over entire training)
            progress = min(step / max(est_total_steps, 1), 1.0)
            lambda_s = lambda_s_final * progress

            # LR schedule: 5% warmup, constant, linear decay last 20%
            warmup_steps = int(0.05 * est_total_steps)
            decay_start = int(0.80 * est_total_steps)
            if step < warmup_steps:
                lr_mult = step / max(warmup_steps, 1)
            elif step >= decay_start:
                decay_progress = (step - decay_start) / max(est_total_steps - decay_start, 1)
                lr_mult = 1.0 - decay_progress
            else:
                lr_mult = 1.0
            current_lr = lr * lr_mult
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # Forward + loss
            optimizer.zero_grad()
            x_hat, z, z_pre = sae(sae_input)
            loss, metrics = sae.compute_loss(
                sae_input, x_hat, z, z_pre, lambda_s, lambda_p, c
            )

            # Backward + clip + step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), grad_clip)
            optimizer.step()

            # Decoder normalization
            sae.normalize_decoder()

            # Track dead features
            dead_tracker.update(z.detach(), step)

            step += 1

            # Logging
            if step % log_every == 0:
                dead_pct = dead_tracker.dead_fraction(step) * 100
                metrics["dead_pct"] = dead_pct
                metrics["lambda_s"] = lambda_s
                metrics["lr"] = current_lr
                metrics["step"] = step
                metrics["progress_pct"] = progress * 100

                log(f"  step {step}/{est_total_steps} ({progress * 100:.1f}%) | "
                    f"mse={metrics['mse']:.4f} | L0={metrics['l0']:.1f} | "
                    f"VE={metrics['var_explained']:.3f} | "
                    f"dead={dead_pct:.1f}% | λ_s={lambda_s:.4f} | "
                    f"lr={current_lr:.2e}")

                training_log.append(metrics)

            # Checkpointing
            closest_ckpt = min(ckpt_steps.keys(), key=lambda s: abs(s - step),
                               default=None)
            if closest_ckpt is not None and abs(step - closest_ckpt) < 2:
                pct = ckpt_steps.pop(closest_ckpt)
                ckpt_path = f"{run_dir}/checkpoint_{pct}pct.pt"
                if not os.path.exists(ckpt_path):
                    _save_checkpoint(
                        ckpt_path, sae, optimizer, normalizer, config,
                        metrics, step, est_total_steps,
                    )
                    vol.commit()
                    log(f"  checkpoint saved: {pct}% ({ckpt_path})")

            # Stop when we've reached estimated total steps
            if step >= est_total_steps:
                break

        # Break outer loop too
        if step >= est_total_steps:
            break

    # --- Final save ---
    final_metrics = training_log[-1] if training_log else {}
    _save_checkpoint(
        final_ckpt, sae, optimizer, normalizer, config,
        final_metrics, step, est_total_steps,
    )

    # Save training log
    with open(f"{run_dir}/training_log.json", "w") as f:
        json.dump(training_log, f)

    vol.commit()

    log(f"{run_name}: TRAINING COMPLETE — {step} steps, "
        f"L0={final_metrics.get('l0', '?')}, "
        f"VE={final_metrics.get('var_explained', '?')}, "
        f"dead={final_metrics.get('dead_pct', '?')}%")

    # Push to HuggingFace
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        repo_id = f"{HF_USERNAME}/scar-weights"
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True,
                        token=hf_token)
        api.upload_file(
            path_or_fileobj=final_ckpt,
            path_in_repo=f"sae/{run_name}/checkpoint_final.pt",
            repo_id=repo_id, repo_type="model", token=hf_token,
            commit_message=f"Upload SAE checkpoint: {run_name}",
        )
        api.upload_file(
            path_or_fileobj=f"{run_dir}/config.json",
            path_in_repo=f"sae/{run_name}/config.json",
            repo_id=repo_id, repo_type="model", token=hf_token,
        )
        log(f"{run_name}: pushed to {repo_id}")
    except Exception as e:
        log(f"{run_name}: HF push failed (non-fatal): {e}")


def _save_checkpoint(path, sae, optimizer, normalizer, config, metrics, step,
                     total_steps):
    """Save a training checkpoint."""
    import torch
    torch.save({
        "sae_state_dict": sae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "normalizer_state_dict": normalizer.state_dict(),
        "config": config,
        "metrics": metrics,
        "step": step,
        "total_steps": total_steps,
    }, path)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5b: Modality-Mixed SAE Training
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={SAVE_DIR: vol},
    secrets=[HF_SECRET],
    timeout=86400,  # 24 hours max
)
def train_sae_mixed(
    layer: int = 19,
    d_sae: int = 16384,
    run_name: str = "mixed_L19_16k",
    lambda_s_final: float = 0.01,
    lambda_p: float = 3e-6,
    lr: float = 2e-4,
    sae_batch_size: int = 4096,
    model_batch_size: int = 8,
    max_seq_len: int = 1024,
    max_contracts: int = None,
    c: float = 4.0,
    grad_clip: float = 1.0,
    init_threshold: float = 0.1,
    log_every: int = 100,
    checkpoint_pcts: list = None,
    code_to_query_ratio: int = 4,
):
    """Train JumpReLU SAE on MIXED modality activations (code + queries).

    Same architecture and training recipe as train_sae(), but uses
    stream_mixed_activations() which interleaves code contract activations
    with vulnerability query/positive text activations.

    This ensures SAE features learn patterns shared across both modalities:
    - Code patterns: reentrancy guards, access control modifiers, etc.
    - Query patterns: vulnerability descriptions, severity indicators
    - Shared patterns: the bridge between "reentrancy" in text and call.value() in code
    """
    import torch
    import torch.nn.functional as F
    import json
    from tqdm import tqdm

    if checkpoint_pcts is None:
        checkpoint_pcts = [25, 50, 75, 100]

    vol.reload()

    run_dir = f"{SAVE_DIR}/{run_name}"
    final_ckpt = f"{run_dir}/checkpoint_final.pt"

    # Skip if already done
    if os.path.exists(final_ckpt):
        log(f"{run_name}: already trained, skipping")
        return

    # Auto-load calibrated lambda from volume if available
    cal_path = f"{SAVE_DIR}/calibration/lambda_s_sweep.json"
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            cal_data = json.load(f)
        if "best_lambda_s" in cal_data:
            calibrated = cal_data["best_lambda_s"]
            log(f"{run_name}: using calibrated lambda_s={calibrated}")
            lambda_s_final = calibrated

    os.makedirs(run_dir, exist_ok=True)

    # Save config
    config = {
        "run_name": run_name, "layer": layer, "d_in": D_IN, "d_sae": d_sae,
        "lambda_s_final": lambda_s_final, "lambda_p": lambda_p, "lr": lr,
        "sae_batch_size": sae_batch_size, "model_batch_size": model_batch_size,
        "max_seq_len": max_seq_len, "c": c, "grad_clip": grad_clip,
        "init_threshold": init_threshold, "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME, "modality": "mixed",
        "code_to_query_ratio": code_to_query_ratio,
        "pairs_dataset": PAIRS_DATASET,
    }
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{run_name}: device={device}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    # Load model (frozen, fp16)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"{run_name}: loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(device).eval()

    for param in model.parameters():
        param.requires_grad = False

    log(f"{run_name}: model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    # Initialize SAE
    JumpReLUSAE = build_sae_class()
    sae = JumpReLUSAE(d_in=D_IN, d_sae=d_sae, init_threshold=init_threshold).to(device)
    sae_params = sum(p.numel() for p in sae.parameters())
    log(f"{run_name}: SAE initialized ({sae_params / 1e6:.1f}M params, {d_sae} features)")

    # Check for resume checkpoint (rolling > percentage-based)
    resume_step = 0
    resume_ckpt = None
    rolling_ckpt_path = f"{run_dir}/checkpoint_rolling.pt"
    if os.path.exists(rolling_ckpt_path):
        resume_ckpt = rolling_ckpt_path
    else:
        for pct in sorted(checkpoint_pcts[:-1], reverse=True):
            ckpt_path = f"{run_dir}/checkpoint_{pct}pct.pt"
            if os.path.exists(ckpt_path):
                resume_ckpt = ckpt_path
                break

    # Optimizer
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    normalizer = ActivationNormalizer(D_IN)
    dead_tracker = DeadFeatureTracker(d_sae)

    if resume_ckpt:
        log(f"{run_name}: resuming from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        sae.load_state_dict(ckpt["sae_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        normalizer.load_state_dict(ckpt["normalizer_state_dict"])
        resume_step = ckpt["step"]
        log(f"{run_name}: resumed at step {resume_step}")

    # Estimate total steps (slightly more than code-only due to query batches)
    # ~231k contracts + ~8k query/positive texts = ~240k texts
    est_total_tokens = 240000 * 1500
    est_total_steps = est_total_tokens // sae_batch_size
    log(f"{run_name}: estimated ~{est_total_steps} steps (~{est_total_tokens / 1e6:.0f}M tokens)")

    # Checkpoint step targets
    ckpt_steps = {int(est_total_steps * pct / 100): pct for pct in checkpoint_pcts}

    hf_token = os.environ.get("HF_TOKEN")

    # --- Training loop (same as train_sae but with mixed streaming) ---
    step = 0
    act_buffer = []
    act_buffer_size = 0
    training_log = []

    log(f"{run_name}: starting MIXED-MODALITY training (layer={layer}, d_sae={d_sae}, "
        f"lambda_s_final={lambda_s_final}, code:query ratio={code_to_query_ratio}:1)")

    for act_batch in stream_mixed_activations(
        model, tokenizer, layer, max_seq_len, model_batch_size, device,
        hf_token=hf_token, max_contracts=max_contracts,
        code_to_query_ratio=code_to_query_ratio,
    ):
        # Accumulate in buffer
        act_buffer.append(act_batch)
        act_buffer_size += act_batch.shape[0]

        # Train when we have enough tokens
        while act_buffer_size >= sae_batch_size:
            all_acts = torch.cat(act_buffer, dim=0)
            sae_input = all_acts[:sae_batch_size]
            remainder = all_acts[sae_batch_size:]
            act_buffer = [remainder] if remainder.shape[0] > 0 else []
            act_buffer_size = remainder.shape[0] if remainder.shape[0] > 0 else 0

            # Shuffle (mixes code and query activations within batch)
            perm = torch.randperm(sae_input.shape[0], device=device)
            sae_input = sae_input[perm]

            # Skip steps if resuming
            if step < resume_step:
                step += 1
                continue

            # Normalize activations
            sae_input = normalizer.normalize(sae_input)

            # Lambda_S warmup (linear over entire training)
            progress = min(step / max(est_total_steps, 1), 1.0)
            lambda_s = lambda_s_final * progress

            # LR schedule: 5% warmup, constant, linear decay last 20%
            warmup_steps = int(0.05 * est_total_steps)
            decay_start = int(0.80 * est_total_steps)
            if step < warmup_steps:
                lr_mult = step / max(warmup_steps, 1)
            elif step >= decay_start:
                decay_progress = (step - decay_start) / max(est_total_steps - decay_start, 1)
                lr_mult = 1.0 - decay_progress
            else:
                lr_mult = 1.0
            current_lr = lr * lr_mult
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # Forward + loss
            optimizer.zero_grad()
            x_hat, z, z_pre = sae(sae_input)
            loss, metrics = sae.compute_loss(
                sae_input, x_hat, z, z_pre, lambda_s, lambda_p, c
            )

            # Backward + clip + step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), grad_clip)
            optimizer.step()

            # Decoder normalization
            sae.normalize_decoder()

            # Track dead features
            dead_tracker.update(z.detach(), step)

            step += 1

            # Logging
            if step % log_every == 0:
                dead_pct = dead_tracker.dead_fraction(step) * 100
                metrics["dead_pct"] = dead_pct
                metrics["lambda_s"] = lambda_s
                metrics["lr"] = current_lr
                metrics["step"] = step
                metrics["progress_pct"] = progress * 100

                log(f"  step {step}/{est_total_steps} ({progress * 100:.1f}%) | "
                    f"mse={metrics['mse']:.4f} | L0={metrics['l0']:.1f} | "
                    f"VE={metrics['var_explained']:.3f} | "
                    f"dead={dead_pct:.1f}% | λ_s={lambda_s:.4f} | "
                    f"lr={current_lr:.2e}")

                training_log.append(metrics)

            # Rolling checkpoint every 3000 steps (survives preemption)
            if step > 0 and step % 3000 == 0:
                latest_metrics = training_log[-1] if training_log else {}
                _save_checkpoint(
                    rolling_ckpt_path, sae, optimizer, normalizer, config,
                    latest_metrics, step, est_total_steps,
                )
                vol.commit()
                log(f"  rolling checkpoint saved at step {step}")

            # Percentage-based checkpointing
            closest_ckpt = min(ckpt_steps.keys(), key=lambda s: abs(s - step),
                               default=None)
            if closest_ckpt is not None and abs(step - closest_ckpt) < 2:
                pct = ckpt_steps.pop(closest_ckpt)
                ckpt_path = f"{run_dir}/checkpoint_{pct}pct.pt"
                if not os.path.exists(ckpt_path):
                    pct_metrics = training_log[-1] if training_log else {}
                    _save_checkpoint(
                        ckpt_path, sae, optimizer, normalizer, config,
                        pct_metrics, step, est_total_steps,
                    )
                    vol.commit()
                    log(f"  checkpoint saved: {pct}% ({ckpt_path})")

            if step >= est_total_steps:
                break

        if step >= est_total_steps:
            break

    # --- Final save ---
    final_metrics = training_log[-1] if training_log else {}
    _save_checkpoint(
        final_ckpt, sae, optimizer, normalizer, config,
        final_metrics, step, est_total_steps,
    )

    # Clean up rolling checkpoint after successful completion
    if os.path.exists(rolling_ckpt_path):
        os.remove(rolling_ckpt_path)

    # Save training log
    with open(f"{run_dir}/training_log.json", "w") as f:
        json.dump(training_log, f)

    vol.commit()

    log(f"{run_name}: TRAINING COMPLETE — {step} steps, "
        f"L0={final_metrics.get('l0', '?')}, "
        f"VE={final_metrics.get('var_explained', '?')}, "
        f"dead={final_metrics.get('dead_pct', '?')}%")

    # Push to HuggingFace
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        repo_id = f"{HF_USERNAME}/scar-weights"
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True,
                        token=hf_token)
        api.upload_file(
            path_or_fileobj=final_ckpt,
            path_in_repo=f"sae/{run_name}/checkpoint_final.pt",
            repo_id=repo_id, repo_type="model", token=hf_token,
            commit_message=f"Upload mixed-modality SAE: {run_name}",
        )
        api.upload_file(
            path_or_fileobj=f"{run_dir}/config.json",
            path_in_repo=f"sae/{run_name}/config.json",
            repo_id=repo_id, repo_type="model", token=hf_token,
        )
        log(f"{run_name}: pushed to {repo_id}")
    except Exception as e:
        log(f"{run_name}: HF push failed (non-fatal): {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Lambda_S Calibration
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={SAVE_DIR: vol},
    secrets=[HF_SECRET],
    timeout=7200,
)
def calibrate_lambda_s(layer: int = 19, d_sae: int = 16384):
    """Quick sweep to find lambda_s that gives L0 ≈ 20.

    Tests lambda_s in {0.005, 0.01, 0.02, 0.05} on ~5% of data.
    Saves results to calibration/lambda_s_sweep.json.
    """
    import torch
    import json

    vol.reload()

    lambda_candidates = [0.1, 0.15, 0.2, 0.3]
    cal_contracts = 12000  # ~5% of dataset
    cal_steps = 1000       # enough to see stable L0 after warmup

    cal_dir = f"{SAVE_DIR}/calibration"
    output_path = f"{cal_dir}/lambda_s_sweep.json"

    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        # Only skip if all current candidates are already covered
        all_covered = all(str(lam) in existing for lam in lambda_candidates)
        if all_covered:
            log("Calibration: already done, skipping")
            log(f"Calibration results: {json.dumps(existing, indent=2)}")
            return existing.get("best_lambda_s")
        else:
            log("Calibration: re-running with new lambda candidates")

    os.makedirs(cal_dir, exist_ok=True)

    device = torch.device("cuda")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("Calibration: loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    hf_token = os.environ.get("HF_TOKEN")

    results = {}

    for lam in lambda_candidates:
        log(f"Calibration: testing lambda_s={lam}")

        JumpReLUSAE = build_sae_class()
        sae = JumpReLUSAE(d_in=D_IN, d_sae=d_sae).to(device)
        optimizer = torch.optim.AdamW(sae.parameters(), lr=2e-4, betas=(0.9, 0.999))
        normalizer = ActivationNormalizer(D_IN)

        step = 0
        l0_history = []
        act_buffer = []
        act_buffer_size = 0

        log(f"  Starting stream_activations for lambda_s={lam}...")
        for act_batch in stream_activations(
            model, tokenizer, layer, 1024, 8, device,
            hf_token=hf_token, max_contracts=cal_contracts,
        ):
            if step == 0 and len(act_buffer) == 0:
                log(f"  First activation batch received: shape={act_batch.shape}, "
                    f"dtype={act_batch.dtype}")
            act_buffer.append(act_batch)
            act_buffer_size += act_batch.shape[0]

            while act_buffer_size >= 4096:
                all_acts = torch.cat(act_buffer, dim=0)
                sae_input = all_acts[:4096]
                remainder = all_acts[4096:]
                act_buffer = [remainder] if remainder.shape[0] > 0 else []
                act_buffer_size = remainder.shape[0] if remainder.shape[0] > 0 else 0

                sae_input = normalizer.normalize(sae_input)

                # Lambda warmup (simulate full warmup compressed to cal_steps)
                progress = min(step / cal_steps, 1.0)
                current_lambda = lam * progress

                optimizer.zero_grad()
                x_hat, z, z_pre = sae(sae_input)
                loss, metrics = sae.compute_loss(
                    sae_input, x_hat, z, z_pre, current_lambda
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                optimizer.step()
                sae.normalize_decoder()

                step += 1
                if step <= 3 or step % 200 == 0:
                    log(f"  cal step {step}/{cal_steps}: loss={loss.item():.4f} "
                        f"mse={metrics.get('mse', 0):.4f} L0={metrics.get('l0', 0):.1f}")
                if step > cal_steps * 0.8:  # only record L0 after warmup
                    l0_history.append(metrics["l0"])

                if step >= cal_steps:
                    break
            if step >= cal_steps:
                break

        avg_l0 = sum(l0_history) / len(l0_history) if l0_history else 0
        results[str(lam)] = {
            "avg_l0": round(avg_l0, 1),
            "final_mse": round(metrics.get("mse", 0), 4),
            "final_ve": round(metrics.get("var_explained", 0), 3),
            "steps": step,
        }
        log(f"  lambda_s={lam} → L0={avg_l0:.1f}, VE={metrics.get('var_explained', 0):.3f}")

        del sae, optimizer
        torch.cuda.empty_cache()

    # Find best lambda
    target_l0 = 20
    best_lambda = min(lambda_candidates,
                      key=lambda lam: abs(results[str(lam)]["avg_l0"] - target_l0))
    results["best_lambda_s"] = best_lambda
    results["target_l0"] = target_l0

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    log(f"Calibration complete. Best lambda_s={best_lambda} "
        f"(L0={results[str(best_lambda)]['avg_l0']})")

    return best_lambda


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: Feature Inspection
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={SAVE_DIR: vol},
    secrets=[HF_SECRET],
    timeout=7200,
)
def inspect_features(run_name: str = "primary_L19_16k", n_features: int = 50,
                     n_examples: int = 20):
    """Inspect top SAE features to verify security-semantic patterns.

    Loads trained SAE, streams contracts, records top activating tokens
    with their surrounding code context.
    """
    import torch
    import json
    from collections import defaultdict

    vol.reload()

    run_dir = f"{SAVE_DIR}/{run_name}"
    ckpt_path = f"{run_dir}/checkpoint_final.pt"
    output_path = f"{SAVE_DIR}/feature_inspection/{run_name}_features.json"

    if os.path.exists(output_path):
        log(f"Inspect: {output_path} exists, skipping")
        return

    if not os.path.exists(ckpt_path):
        log(f"Inspect: no checkpoint at {ckpt_path}")
        return

    os.makedirs(f"{SAVE_DIR}/feature_inspection", exist_ok=True)

    device = torch.device("cuda")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    # Load SAE
    JumpReLUSAE = build_sae_class()
    sae = JumpReLUSAE(
        d_in=config["d_in"], d_sae=config["d_sae"],
        init_threshold=config.get("init_threshold", 0.1)
    ).to(device)
    sae.load_state_dict(ckpt["sae_state_dict"])
    sae.eval()

    normalizer = ActivationNormalizer(config["d_in"])
    if "normalizer_state_dict" in ckpt:
        normalizer.load_state_dict(ckpt["normalizer_state_dict"])

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("Inspect: loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    hf_token = os.environ.get("HF_TOKEN")
    layer = config["layer"]

    # Track top activations per feature
    # feature_id -> list of (activation_value, contract_idx, token_pos, context)
    feature_tops = defaultdict(list)
    feature_fire_count = torch.zeros(config["d_sae"], device=device)
    total_tokens = 0
    inspect_contracts = 10000

    log(f"Inspect: streaming {inspect_contracts} contracts...")

    from datasets import load_dataset
    ds = load_dataset(DATASET_NAME, split="train", token=hf_token)
    ds = ds.select(range(min(inspect_contracts, len(ds))))

    contract_texts = [row["contract_code"] for row in ds]

    for batch_start in range(0, len(contract_texts), 4):
        batch_codes = contract_texts[batch_start:batch_start + 4]

        # Tokenize
        enc = tokenizer(
            batch_codes, return_tensors="pt", truncation=True,
            max_length=1024, padding=True,
        ).to(device)

        # Forward with hook
        activations_captured = []

        def hook_fn(module, input, output):
            activations_captured.append(output[0].detach().float())

        handle = model.model.layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**enc)
        handle.remove()

        if not activations_captured:
            continue

        acts = activations_captured[0]  # (batch, seq_len, d_in)

        # Normalize and run through SAE
        flat_acts = acts.reshape(-1, D_IN)
        flat_acts = normalizer.normalize(flat_acts)

        with torch.no_grad():
            z, _ = sae.encode(flat_acts)

        # Track fire counts
        feature_fire_count += (z > 0).float().sum(dim=0)
        total_tokens += z.shape[0]

        # Record top activations for each feature
        for feat_idx in range(min(config["d_sae"], 1000)):  # check top 1000
            feat_acts = z[:, feat_idx]
            if feat_acts.max() > 0:
                top_val, top_pos = feat_acts.max(dim=0)
                # Map flat position back to (batch_idx, token_pos)
                seq_len = acts.shape[1]
                batch_idx = top_pos.item() // seq_len
                token_pos = top_pos.item() % seq_len
                actual_contract_idx = batch_start + batch_idx

                # Get context (decode surrounding tokens)
                if batch_idx < len(batch_codes):
                    code = batch_codes[batch_idx]
                    # Approximate character position from token position
                    tokens = tokenizer.encode(code, add_special_tokens=False)
                    if token_pos < len(tokens):
                        prefix = tokenizer.decode(tokens[max(0, token_pos - 10):token_pos])
                        center = tokenizer.decode(tokens[token_pos:token_pos + 1])
                        suffix = tokenizer.decode(tokens[token_pos + 1:token_pos + 11])
                        context = f"...{prefix}>>>{center}<<<{suffix}..."
                    else:
                        context = ""
                else:
                    context = ""

                feature_tops[feat_idx].append({
                    "activation": round(top_val.item(), 3),
                    "contract_idx": actual_contract_idx,
                    "token_pos": token_pos,
                    "context": context[:200],
                })

        del activations_captured, acts, z
        torch.cuda.empty_cache()

    # Build report for top features by activation frequency
    # Only consider features we tracked examples for (first 1000)
    feature_freq = feature_fire_count / max(total_tokens, 1)
    tracked_freq = feature_freq[:1000]
    top_feature_indices = tracked_freq.argsort(descending=True)[:n_features].cpu().tolist()

    security_keywords = [
        "call.value", "delegatecall", "selfdestruct", "tx.origin", "block.timestamp",
        "msg.sender", "transfer", "approve", "allowance", "balanceOf", "reentrancy",
        "require(", "assert(", "revert", "overflow", "underflow", "onlyOwner",
        "modifier", "external", "payable", "fallback", "receive",
    ]

    report = {
        "run_name": run_name,
        "total_tokens_inspected": total_tokens,
        "n_contracts": inspect_contracts,
        "features": {},
    }

    for feat_idx in top_feature_indices:
        freq = feature_freq[feat_idx].item()
        examples = sorted(feature_tops.get(feat_idx, []),
                          key=lambda x: x["activation"], reverse=True)[:n_examples]

        # Check for security keywords in contexts
        all_context = " ".join(e["context"] for e in examples)
        found_keywords = [kw for kw in security_keywords if kw in all_context]

        report["features"][str(feat_idx)] = {
            "activation_frequency": round(freq, 6),
            "mean_activation": round(
                sum(e["activation"] for e in examples) / max(len(examples), 1), 3
            ),
            "top_examples": examples,
            "security_keywords_found": found_keywords,
        }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    vol.commit()

    # Summary
    features_with_security = sum(
        1 for f in report["features"].values() if f["security_keywords_found"]
    )
    log(f"Inspect: saved {n_features} features to {output_path}")
    log(f"  {features_with_security}/{n_features} features have security keywords")

    # Push to HF
    try:
        from huggingface_hub import HfApi
        hf_token = os.environ.get("HF_TOKEN")
        api = HfApi(token=hf_token)
        repo_id = f"{HF_USERNAME}/scar-weights"
        api.upload_file(
            path_or_fileobj=output_path,
            path_in_repo=f"sae/{run_name}/feature_inspection.json",
            repo_id=repo_id, repo_type="model", token=hf_token,
        )
        log(f"Inspect: pushed to {repo_id}")
    except Exception as e:
        log(f"Inspect: HF push failed (non-fatal): {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(mode: str = "primary", layer: int = 19, width: int = 16384):
    """Run SAE pretraining pipeline.

    Args:
        mode: "calibrate" | "primary" | "ablations" | "all" | "inspect" | "mixed"
        layer: target layer (for single runs)
        width: SAE width (for single runs)
    """
    import json
    from datetime import datetime

    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"SCAR SAE Pretraining (mode={mode})")

    # Calibration result — train_sae auto-loads from volume if available,
    # but we also capture it here when running calibration in this session
    best_lambda = 0.01  # default fallback

    if mode == "calibrate":
        print("--- CALIBRATION ---")
        result = calibrate_lambda_s.remote(layer=19, d_sae=16384)
        if result is not None:
            print(f"Calibration result: best lambda_s = {result}")
        return

    if mode == "all":
        print("--- CALIBRATION (part of all) ---")
        result = calibrate_lambda_s.remote(layer=19, d_sae=16384)
        if result is not None:
            best_lambda = result
            print(f"Using calibrated lambda_s={best_lambda}")
        # Note: train_sae also auto-reads calibration from volume

    if mode in ("primary", "all"):
        print(f"\n--- PRIMARY: Layer {layer}, Width {width} ---")
        run_name = f"primary_L{layer}_{width // 1024}k"
        train_sae.remote(
            layer=layer, d_sae=width, run_name=run_name,
            lambda_s_final=best_lambda,
        )

    if mode in ("ablations", "all"):
        print("\n--- ABLATIONS: Layer sweep (L14, L24 parallel) ---")
        h1 = train_sae.spawn(layer=14, d_sae=16384,
                              run_name="ablation_L14_16k",
                              lambda_s_final=best_lambda)
        h2 = train_sae.spawn(layer=24, d_sae=16384,
                              run_name="ablation_L24_16k",
                              lambda_s_final=best_lambda)
        h1.get()
        print("  L14 ablation complete")
        h2.get()
        print("  L24 ablation complete")

        print("\n--- ABLATIONS: Width sweep (8k, 32k parallel) ---")
        h3 = train_sae.spawn(layer=19, d_sae=8192,
                              run_name="ablation_L19_8k",
                              lambda_s_final=best_lambda)
        h4 = train_sae.spawn(layer=19, d_sae=32768,
                              run_name="ablation_L19_32k",
                              lambda_s_final=best_lambda)
        h3.get()
        print("  8k ablation complete")
        h4.get()
        print("  32k ablation complete")

    if mode in ("inspect", "all"):
        print("\n--- FEATURE INSPECTION ---")
        run_name = f"primary_L{layer}_{width // 1024}k"
        inspect_features.remote(run_name=run_name)

    if mode == "mixed":
        # Modality-mixed SAE: trains on both code + query activations.
        # The critical fix for the interpretability failure — current SAE only
        # saw code tokens, so it learned generic code features. This version
        # sees vulnerability descriptions too, learning shared features.
        print(f"\n--- MIXED-MODALITY SAE: Layer {layer}, Width {width} ---")
        run_name = f"mixed_L{layer}_{width // 1024}k"
        train_sae_mixed.remote(
            layer=layer, d_sae=width, run_name=run_name,
            lambda_s_final=best_lambda,
            code_to_query_ratio=4,
        )

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
