"""
SCAR — minimal weight loading and inspection example.

Downloads the published SCAR weights from HuggingFace and prints the structure
of every component (frozen SAE, backbone LoRA, SAE-LoRA, IDF weights, training
config). Use this as a first-step sanity check that all components downloaded
and load correctly.

The full retrieval encoder — handling bidirectional attention monkey-patching,
activation normalization (target_norm ≈ 39.19), per-token TopK on z_pre, sum
pooling with log1p saturation, IDF weighting, and document-level TopK — lives
in `scripts/step9_evaluation.py`. That implementation is what produced the
paper numbers and is what you should call for any real retrieval workload.

Usage:
    python examples/encode_and_retrieve.py
    python examples/encode_and_retrieve.py --variant scar-15ep
"""

import argparse
import torch
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["scar-25ep", "scar-15ep"],
        default="scar-25ep",
        help="Which retrieval checkpoint to inspect (default: scar-25ep)",
    )
    args = parser.parse_args()

    print(f"Downloading SCAR ({args.variant}) weights from HuggingFace...")
    local_dir = snapshot_download("Farseen0/scar-weights")
    print(f"  cached at: {local_dir}\n")

    # ----- Frozen JumpReLU SAE -----
    sae_path = f"{local_dir}/sae/checkpoint_final.pt"
    sae_ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
    sae_state = sae_ckpt["sae_state_dict"]

    print("=== Frozen JumpReLU SAE ===")
    for k, v in sae_state.items():
        print(f"  {k:<18s}  shape={tuple(v.shape)}")
    d_in = sae_state["W_enc"].shape[0]
    d_sae = sae_state["W_enc"].shape[1]
    print(f"  d_in       = {d_in}")
    print(f"  d_sae      = {d_sae}")
    print(f"  expansion  = {d_sae / d_in:.1f}x")
    if "normalizer_state_dict" in sae_ckpt:
        norm = sae_ckpt["normalizer_state_dict"]
        print(f"  target_norm = {norm.get('target_norm', 'n/a')}  "
              f"(activations are normalized to this scale before SAE encode)")
    print()

    # ----- Retrieval checkpoint (LoRA + SAE-LoRA + IDF + config) -----
    retr_path = f"{local_dir}/{args.variant}/checkpoint_final.pt"
    retr_ckpt = torch.load(retr_path, map_location="cpu", weights_only=False)

    print(f"=== {args.variant} retrieval checkpoint ===")
    print(f"  top-level keys: {list(retr_ckpt.keys())}\n")

    # SAE-LoRA
    sae_lora = retr_ckpt["sae_lora_state"]
    print("=== SAE-LoRA (rank-r adaptation of SAE encoder W_enc) ===")
    for k, v in sae_lora.items():
        print(f"  {k:<10s}  shape={tuple(v.shape)}")
    A = sae_lora["lora_A"]
    B = sae_lora["lora_B"]
    print(f"  effective rank: {A.shape[1]}")
    print(f"  added params:   {A.numel() + B.numel():,} "
          f"({100 * (A.numel() + B.numel()) / (d_in * d_sae):.2f}% of W_enc)")
    print()

    # IDF
    idf = retr_ckpt["idf_weights"]
    print("=== IDF weights (applied after sum-pooling, before doc-level TopK) ===")
    print(f"  shape:  {tuple(idf.shape)}")
    print(f"  active: {(idf > 0).sum().item():,} / {idf.numel():,} features")
    print(f"  range:  [{idf.min():.4f}, {idf.max():.4f}]   mean={idf.mean():.4f}")
    print()

    # Training config
    cfg = retr_ckpt["config"]
    print("=== Training config ===")
    for key in [
        "run_name", "epochs", "batch_size", "lr", "temperature",
        "lora_rank", "sae_lora_rank", "topk_query", "topk_doc",
        "max_seq_len", "model_name", "dataset_name",
    ]:
        if key in cfg:
            print(f"  {key:<16s} = {cfg[key]}")
    print()

    print("All SCAR components loaded and verified.")
    print()
    print("Next steps:")
    print("  - For paper-faithful retrieval: see scripts/step9_evaluation.py")
    print("  - For training: see scripts/step6_retrieval.py")
    print("  - For SAE pretraining: see scripts/step5_sae.py")


if __name__ == "__main__":
    main()
