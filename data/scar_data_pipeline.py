"""
SCAR Data Pipeline — Eval Expansion + Synthetic Generation
==========================================================
Builds the SCAR-branded datasets:
  1. scar-eval:   500+ multi-source eval pairs (expanded from 91)
  2. scar-pairs:  Real + synthetic training pairs (~20-25k target)
  3. scar-corpus: Reuse existing 231k contracts

Modes:
  modal run scripts/scar_data_pipeline.py --mode eval_expand    # Build scar-eval (500+)
  modal run scripts/scar_data_pipeline.py --mode synthetic      # Generate synthetic pairs
  modal run scripts/scar_data_pipeline.py --mode merge_push     # Merge all + push to HF
  modal run scripts/scar_data_pipeline.py --mode check          # Check dataset stats

Requirements:
  - Modal secret: huggingface-token
  - Modal secret: aws-credentials (for synthetic generation only)
"""

import modal
import os
import json
import hashlib
import random
from datetime import datetime

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("scar-data-pipeline")

HF_SECRET = modal.Secret.from_name("huggingface-token")
HF_USERNAME = "Farseen0"

VOLUME_NAME = "scar-data-vol"
DATA_DIR = "/scar_data"

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.19.0",
        "huggingface_hub>=0.23.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
        "rank_bm25>=0.2.2",
    )
    .apt_install("git")
)

image_with_aws = image.pip_install("boto3>=1.34.0")


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def normalize_solidity(source: str) -> str:
    """Strip comments, pragmas, whitespace for dedup."""
    import re
    text = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(pragma|// SPDX).*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def hash_contract(source: str) -> str:
    """SHA256 hash of normalized source."""
    if not source or not str(source).strip():
        return ""
    normalized = normalize_solidity(str(source))
    return hashlib.sha256(normalized.encode()).hexdigest()


def build_query(severity: str, title: str, description: str) -> str:
    """Standard SCAR query format."""
    sev = str(severity).strip().upper()
    if sev == "CRITICAL":
        sev = "HIGH"
    desc_trunc = str(description)[:300].strip()
    if len(str(description)) > 300:
        desc_trunc += "..."
    return f"{sev} severity: {title}. {desc_trunc}"


def extract_severity(raw) -> str:
    """Normalize severity from various formats."""
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    s = str(raw).strip().upper()
    for valid in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"):
        if valid in s:
            return valid
    return s


# ---------------------------------------------------------------------------
# Step 1: Eval Expansion — Build scar-eval (500+ pairs)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes={DATA_DIR: vol},
    timeout=3600,
    memory=16384,
)
def build_eval_set():
    """
    Build expanded eval set from multiple sources:
    1. Existing 91 FORGE-Curated eval pairs (keep all)
    2. FORGE-Artifacts unused findings (not in training)
    3. Holdout newest 10% from training sources
    """
    import pandas as pd
    from datasets import load_dataset, Dataset
    from collections import defaultdict
    import subprocess

    token = os.environ.get("HF_TOKEN")
    os.makedirs(DATA_DIR, exist_ok=True)

    # ===== Load existing eval (91 pairs) =====
    log("Loading existing eval set...")
    old_eval = load_dataset(f"{HF_USERNAME}/scar-eval", split="train", token=token)
    old_eval_df = old_eval.to_pandas()
    log(f"Existing eval: {len(old_eval_df)} pairs")

    # ===== Load existing training pairs (to know what to exclude) =====
    log("Loading existing training pairs...")
    old_pairs = load_dataset(f"{HF_USERNAME}/scar-pairs", split="train", token=token)
    old_pairs_df = old_pairs.to_pandas()
    log(f"Existing training: {len(old_pairs_df)} pairs")

    # Build hash set of all training positives + eval ground truths
    training_pos_hashes = set()
    for code in old_pairs_df["positive"]:
        h = hash_contract(str(code))
        if h:
            training_pos_hashes.add(h)

    eval_gt_hashes = set()
    for code in old_eval_df["ground_truth_code"]:
        h = hash_contract(str(code))
        if h:
            eval_gt_hashes.add(h)

    log(f"Training positive hashes: {len(training_pos_hashes)}")
    log(f"Eval ground truth hashes: {len(eval_gt_hashes)}")

    new_eval_records = []

    # ===== Source 1: FORGE-Artifacts (unused findings) =====
    log("\n--- FORGE-Artifacts: extracting unused findings for eval ---")
    artifacts_dir = f"{DATA_DIR}/FORGE-Artifacts"
    if not os.path.exists(artifacts_dir):
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/shenyimings/FORGE-Artifacts.git", artifacts_dir],
            check=True,
        )

    # Build contract index
    import glob
    contract_index = {}  # lowercase filename → path
    sol_files = glob.glob(f"{artifacts_dir}/**/*.sol", recursive=True)
    for fp in sol_files:
        basename = os.path.basename(fp).lower()
        contract_index[basename] = fp
        # Also index by relative path
        rel = os.path.relpath(fp, artifacts_dir).lower()
        contract_index[rel] = fp
    log(f"FORGE-Artifacts: indexed {len(sol_files)} .sol files")

    # Load findings
    findings_file = f"{artifacts_dir}/FORGE-Artifacts_findings.json"
    if os.path.exists(findings_file):
        with open(findings_file, "r") as f:
            all_findings = json.load(f)
        log(f"FORGE-Artifacts: {len(all_findings)} total findings")
    else:
        # Try alternative structure
        all_findings = []
        for jf in glob.glob(f"{artifacts_dir}/**/findings*.json", recursive=True):
            with open(jf, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_findings.extend(data)
                elif isinstance(data, dict) and "findings" in data:
                    all_findings.extend(data["findings"])
        log(f"FORGE-Artifacts: {len(all_findings)} findings from json files")

    def find_contract_code(finding):
        """Try to find code for a finding."""
        # Method 1: affected_files
        affected = finding.get("affected_files", {})
        if affected and isinstance(affected, dict):
            code_parts = [v for v in affected.values() if v and len(str(v)) > 50]
            if code_parts:
                return "\n".join(code_parts)

        # Method 2: files field
        files_refs = finding.get("files", [])
        if isinstance(files_refs, str):
            files_refs = [files_refs]
        for ref in (files_refs or []):
            ref_clean = str(ref).split("::")[0].split("#")[0].strip()
            basename = os.path.basename(ref_clean).lower()
            if basename in contract_index:
                try:
                    with open(contract_index[basename], "r", errors="replace") as f:
                        return f.read()
                except Exception:
                    pass

        # Method 3: location field
        location = finding.get("location", "")
        if location:
            loc_clean = str(location).split("::")[0].split("#")[0].strip()
            basename = os.path.basename(loc_clean).lower()
            if basename in contract_index:
                try:
                    with open(contract_index[basename], "r", errors="replace") as f:
                        return f.read()
                except Exception:
                    pass

        return ""

    artifacts_eval_count = 0
    seen_queries = set()

    for finding in all_findings:
        severity = extract_severity(finding.get("severity", ""))
        if severity not in ("CRITICAL", "HIGH", "MEDIUM"):
            continue

        title = finding.get("title", "") or finding.get("name", "")
        description = finding.get("description", "") or finding.get("content", "")
        if not title or not description:
            continue

        code = find_contract_code(finding)
        if not code or len(code.strip()) < 100:
            continue

        code_hash = hash_contract(code)
        if not code_hash:
            continue

        # Skip if this code is already in training or old eval
        if code_hash in training_pos_hashes or code_hash in eval_gt_hashes:
            continue

        query = build_query(severity, title, description)
        query_hash = hashlib.sha256(query.encode()).hexdigest()

        # Skip duplicate queries
        if query_hash in seen_queries:
            continue
        seen_queries.add(query_hash)

        sev_upper = severity.upper()
        if sev_upper == "CRITICAL":
            sev_upper = "HIGH"

        # Extract vuln type
        vuln_type = ""
        category = finding.get("category", {})
        if category and isinstance(category, dict):
            cwes = []
            for level_cwes in category.values():
                if isinstance(level_cwes, list):
                    cwes.extend(level_cwes)
            vuln_type = "|".join(cwes[:3])

        new_eval_records.append({
            "query": query,
            "ground_truth_code": code,
            "severity": sev_upper,
            "vuln_type": vuln_type,
            "report_name": finding.get("project_name", "") or finding.get("report", ""),
            "audit_firm": finding.get("firm", "") or finding.get("auditor", ""),
            "report_date": finding.get("date", "") or finding.get("audit_date", ""),
            "source": "FORGE-Artifacts",
        })
        artifacts_eval_count += 1

    log(f"FORGE-Artifacts eval: {artifacts_eval_count} new findings")

    # ===== Source 2: Holdout from training sources =====
    log("\n--- Holding out from training sources for eval ---")

    # Group training pairs by source
    source_groups = defaultdict(list)
    for _, row in old_pairs_df.iterrows():
        source_groups[row.get("source", "unknown")].append(row)

    holdout_count = 0
    remaining_training = []

    for source, pairs in source_groups.items():
        n_holdout = max(1, int(len(pairs) * 0.10))  # 10% holdout

        # Sort by some criterion if possible, otherwise random
        random.seed(42)
        shuffled = list(pairs)
        random.shuffle(shuffled)

        holdout = shuffled[:n_holdout]
        keep = shuffled[n_holdout:]
        remaining_training.extend(keep)

        for row in holdout:
            # Convert training pair to eval format
            code = str(row["positive"])
            code_hash = hash_contract(code)
            if code_hash in eval_gt_hashes:
                remaining_training.append(row)  # Don't duplicate in eval
                continue

            eval_gt_hashes.add(code_hash)  # Track to prevent future dupes

            new_eval_records.append({
                "query": str(row["query"]),
                "ground_truth_code": code,
                "severity": str(row.get("severity", "")),
                "vuln_type": str(row.get("vuln_type", "")),
                "report_name": "",
                "audit_firm": "",
                "report_date": "",
                "source": f"holdout_{source}",
            })
            holdout_count += 1

        log(f"  {source}: {len(pairs)} pairs → {n_holdout} held out, {len(keep)} kept")

    log(f"Total holdout for eval: {holdout_count}")

    # ===== Source 3: EVuLLM unused (if any) =====
    log("\n--- Checking EVuLLM for additional eval data ---")
    try:
        evullm = load_dataset("EVuLLM/defihack", split="train", token=token)
        evullm_count = 0
        for row in evullm:
            code = str(row.get("vulnerable_code_snippet", "") or row.get("code", ""))
            if not code or len(code.strip()) < 100:
                continue

            code_hash = hash_contract(code)
            if code_hash in training_pos_hashes or code_hash in eval_gt_hashes:
                continue

            severity = extract_severity(row.get("severity", "MEDIUM"))
            if severity not in ("CRITICAL", "HIGH", "MEDIUM"):
                continue

            title = str(row.get("title", "") or row.get("vulnerability_type", ""))
            desc = str(row.get("description", "") or row.get("explanation", ""))
            if not title:
                continue

            query = build_query(severity, title, desc)
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            if query_hash in seen_queries:
                continue
            seen_queries.add(query_hash)
            eval_gt_hashes.add(code_hash)

            sev_upper = severity.upper()
            if sev_upper == "CRITICAL":
                sev_upper = "HIGH"

            new_eval_records.append({
                "query": query,
                "ground_truth_code": code,
                "severity": sev_upper,
                "vuln_type": str(row.get("vulnerability_type", "")),
                "report_name": "",
                "audit_firm": "",
                "report_date": "",
                "source": "EVuLLM_holdout",
            })
            evullm_count += 1

        log(f"EVuLLM additional eval: {evullm_count}")
    except Exception as e:
        log(f"EVuLLM load failed: {e}")

    # ===== Combine all eval records =====
    log("\n--- Combining eval set ---")

    # Start with old eval
    combined_eval = []
    for _, row in old_eval_df.iterrows():
        record = row.to_dict()
        record["source"] = record.get("source", "FORGE-Curated")
        combined_eval.append(record)

    # Add new records
    combined_eval.extend(new_eval_records)

    eval_df = pd.DataFrame(combined_eval)
    log(f"\nFINAL EVAL SET: {len(eval_df)} pairs")
    log(f"  By source:")
    for src, count in eval_df["source"].value_counts().items():
        log(f"    {src}: {count}")
    log(f"  By severity:")
    for sev, count in eval_df["severity"].value_counts().items():
        log(f"    {sev}: {count}")

    # Save locally
    eval_path = f"{DATA_DIR}/scar_eval.parquet"
    eval_df.to_parquet(eval_path, index=False)
    log(f"Saved to {eval_path}")

    # Save remaining training pairs (with holdout removed)
    remaining_df = pd.DataFrame(remaining_training)
    remaining_path = f"{DATA_DIR}/scar_pairs_real.parquet"
    remaining_df.to_parquet(remaining_path, index=False)
    log(f"Remaining training pairs (holdout removed): {len(remaining_df)}")
    log(f"Saved to {remaining_path}")

    vol.commit()

    # Push to HF
    log("\nPushing scar-eval to HuggingFace...")
    eval_ds = Dataset.from_pandas(eval_df)
    eval_ds.push_to_hub(
        f"{HF_USERNAME}/scar-eval",
        token=token,
        private=True,
    )
    log(f"Pushed {len(eval_df)} eval pairs to {HF_USERNAME}/scar-eval")

    return {
        "total_eval": len(eval_df),
        "old_eval": len(old_eval_df),
        "forge_artifacts": artifacts_eval_count,
        "holdout": holdout_count,
        "remaining_training": len(remaining_df),
    }


# ---------------------------------------------------------------------------
# Step 2: Synthetic Generation via DeepSeek V3.2 on Bedrock
# ---------------------------------------------------------------------------
@app.function(
    image=image_with_aws,
    secrets=[HF_SECRET],
    volumes={DATA_DIR: vol},
    timeout=7200,
    memory=16384,
)
def generate_synthetic_pairs(
    n_contracts: int = 5000,
    generations_per_contract: int = 3,
    batch_size: int = 50,
):
    """
    Generate synthetic vulnerability pairs using DeepSeek V3.2 on AWS Bedrock.

    Pipeline:
    1. Sample contracts from scar-corpus (Stream 1)
    2. For each contract, ask DeepSeek to identify vulnerabilities (3x)
    3. Self-consistency vote: keep only findings identified 2/3 times
    4. Code grounding: verify LLM-cited code exists in the contract
    5. Build pairs with BM25-mined hard negatives
    """
    import boto3
    import pandas as pd
    from datasets import load_dataset, Dataset
    from rank_bm25 import BM25Okapi
    from tqdm import tqdm

    token = os.environ.get("HF_TOKEN")
    vol.reload()

    # Load scar-corpus
    log("Loading scar-corpus for synthetic generation...")
    corpus = load_dataset(f"{HF_USERNAME}/scar-corpus", split="train", token=token)
    log(f"Corpus: {len(corpus)} contracts")

    # Load eval hashes (to prevent leakage)
    eval_path = f"{DATA_DIR}/scar_eval.parquet"
    if os.path.exists(eval_path):
        eval_df = pd.read_parquet(eval_path)
        eval_hashes = {hash_contract(str(c)) for c in eval_df["ground_truth_code"] if c}
    else:
        eval_hashes = set()
    log(f"Eval hashes to exclude: {len(eval_hashes)}")

    # Sample contracts with vulnerability labels preferred
    log(f"Sampling {n_contracts} contracts...")
    labeled = [i for i in range(len(corpus)) if corpus[i].get("has_vuln_labels")]
    unlabeled = [i for i in range(len(corpus)) if not corpus[i].get("has_vuln_labels")]

    random.seed(42)
    # Prefer labeled contracts (higher quality signal)
    n_labeled = min(len(labeled), n_contracts // 2)
    n_unlabeled = n_contracts - n_labeled
    sampled_indices = random.sample(labeled, n_labeled) + random.sample(unlabeled, min(len(unlabeled), n_unlabeled))
    random.shuffle(sampled_indices)
    log(f"Sampled: {n_labeled} labeled + {len(sampled_indices) - n_labeled} unlabeled")

    # Init Bedrock client
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    SYSTEM_PROMPT = """You are a smart contract security auditor. Analyze the given Solidity contract and identify vulnerabilities.

For each vulnerability found, respond in this exact JSON format:
{
  "vulnerabilities": [
    {
      "severity": "HIGH" or "MEDIUM",
      "title": "Brief vulnerability title",
      "description": "Detailed description of the vulnerability and its impact",
      "vulnerable_code": "The exact code snippet from the contract that is vulnerable (copy verbatim)",
      "vuln_type": "e.g., reentrancy, access-control, integer-overflow, etc."
    }
  ]
}

Rules:
- Only report HIGH or MEDIUM severity vulnerabilities
- The vulnerable_code MUST be an exact substring of the input contract
- Focus on real security issues, not style/gas optimization
- If no vulnerabilities found, return {"vulnerabilities": []}
"""

    def call_deepseek(contract_code: str) -> dict:
        """Call DeepSeek V3.2 on Bedrock."""
        try:
            # Truncate very long contracts
            if len(contract_code) > 12000:
                contract_code = contract_code[:12000] + "\n// ... truncated ..."

            response = bedrock.invoke_model(
                modelId="us.deepseek.r1-v3-0324",  # DeepSeek V3.2 on Bedrock
                body=json.dumps({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Analyze this Solidity contract:\n\n```solidity\n{contract_code}\n```"},
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.3,
                }),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from response
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[\s\S]*"vulnerabilities"[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            return {"vulnerabilities": []}
        except Exception as e:
            log(f"Bedrock call failed: {e}")
            return {"vulnerabilities": []}

    def verify_code_grounding(vuln: dict, contract: str) -> bool:
        """Verify that vulnerable_code is an actual substring of the contract."""
        cited_code = vuln.get("vulnerable_code", "").strip()
        if not cited_code or len(cited_code) < 20:
            return False
        # Normalize whitespace for matching
        import re
        norm_cited = re.sub(r'\s+', ' ', cited_code).strip()
        norm_contract = re.sub(r'\s+', ' ', contract).strip()
        return norm_cited in norm_contract

    def self_consistency_vote(results: list) -> list:
        """Keep vulnerabilities identified in >= 2 of 3 generations."""
        from collections import Counter
        # Key by (title_normalized, severity)
        votes = Counter()
        vuln_by_key = {}

        for result in results:
            for vuln in result.get("vulnerabilities", []):
                key = (vuln.get("title", "").lower().strip(), vuln.get("severity", "").upper())
                votes[key] += 1
                if key not in vuln_by_key:
                    vuln_by_key[key] = vuln

        # Keep those with 2+ votes
        consistent = []
        for key, count in votes.items():
            if count >= 2:
                consistent.append(vuln_by_key[key])
        return consistent

    # ===== Generate pairs =====
    synthetic_pairs = []
    all_positives = []  # For BM25 hard negative mining later

    log(f"\nGenerating synthetic pairs from {len(sampled_indices)} contracts...")
    for batch_start in tqdm(range(0, len(sampled_indices), batch_size)):
        batch_indices = sampled_indices[batch_start:batch_start + batch_size]

        for idx in batch_indices:
            contract = corpus[idx]
            code = contract["contract_code"]
            code_hash = hash_contract(code)

            # Skip if code is in eval set
            if code_hash in eval_hashes:
                continue

            # 3x generation for self-consistency
            results = []
            for _ in range(generations_per_contract):
                result = call_deepseek(code)
                results.append(result)

            # Self-consistency vote
            consistent_vulns = self_consistency_vote(results)

            for vuln in consistent_vulns:
                # Code grounding check
                if not verify_code_grounding(vuln, code):
                    continue

                severity = vuln.get("severity", "MEDIUM").upper()
                if severity == "CRITICAL":
                    severity = "HIGH"
                if severity not in ("HIGH", "MEDIUM"):
                    continue

                query = build_query(severity, vuln["title"], vuln.get("description", ""))
                positive = code  # Full contract as positive

                synthetic_pairs.append({
                    "query": query,
                    "positive": positive,
                    "hard_negative": "",  # Will be filled by BM25 mining
                    "negative_type": "pending_bm25",
                    "source": "synthetic_deepseek",
                    "severity": severity,
                    "vuln_type": vuln.get("vuln_type", ""),
                    "quality_tier": 3,
                })
                all_positives.append(positive)

        log(f"  Batch {batch_start}: {len(synthetic_pairs)} pairs so far")

        # Save checkpoint every 500 contracts
        if (batch_start + batch_size) % 500 == 0:
            checkpoint_df = pd.DataFrame(synthetic_pairs)
            checkpoint_df.to_parquet(f"{DATA_DIR}/synthetic_checkpoint.parquet", index=False)
            vol.commit()

    log(f"\nSynthetic generation complete: {len(synthetic_pairs)} pairs (before BM25 mining)")

    # ===== BM25 Hard Negative Mining =====
    if synthetic_pairs and all_positives:
        log("Mining BM25 hard negatives...")

        # Tokenize positives for BM25
        tokenized = [normalize_solidity(p).split() for p in all_positives]
        bm25 = BM25Okapi(tokenized)

        for i, pair in enumerate(tqdm(synthetic_pairs)):
            query_tokens = normalize_solidity(pair["query"]).split()
            scores = bm25.get_scores(query_tokens)
            # Get top-10 results excluding self
            ranked = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)

            for j in ranked[:10]:
                if j != i and all_positives[j] != pair["positive"]:
                    pair["hard_negative"] = all_positives[j]
                    pair["negative_type"] = "bm25_mined"
                    break

    # Dedup by (query_hash, positive_hash)
    seen = set()
    deduped = []
    for pair in synthetic_pairs:
        key = (
            hashlib.sha256(pair["query"].encode()).hexdigest()[:16],
            hash_contract(pair["positive"])[:16],
        )
        if key not in seen:
            seen.add(key)
            deduped.append(pair)

    log(f"After dedup: {len(deduped)} synthetic pairs")

    # Save
    synth_df = pd.DataFrame(deduped)
    synth_path = f"{DATA_DIR}/scar_pairs_synthetic.parquet"
    synth_df.to_parquet(synth_path, index=False)
    vol.commit()
    log(f"Saved to {synth_path}")

    return {"synthetic_pairs": len(deduped)}


# ---------------------------------------------------------------------------
# Step 3: Merge and Push
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes={DATA_DIR: vol},
    timeout=1800,
    memory=16384,
)
def merge_and_push():
    """Merge real + synthetic pairs, push scar-pairs and scar-corpus to HF."""
    import pandas as pd
    from datasets import load_dataset, Dataset

    token = os.environ.get("HF_TOKEN")
    vol.reload()

    # ===== Load components =====
    real_path = f"{DATA_DIR}/scar_pairs_real.parquet"
    synth_path = f"{DATA_DIR}/scar_pairs_synthetic.parquet"
    eval_path = f"{DATA_DIR}/scar_eval.parquet"

    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Run eval_expand first: {real_path} not found")

    real_df = pd.read_parquet(real_path)
    log(f"Real pairs: {len(real_df)}")

    if os.path.exists(synth_path):
        synth_df = pd.read_parquet(synth_path)
        log(f"Synthetic pairs: {len(synth_df)}")
    else:
        synth_df = pd.DataFrame()
        log("No synthetic pairs found (run synthetic mode first)")

    eval_df = pd.read_parquet(eval_path)
    log(f"Eval pairs: {len(eval_df)}")

    # ===== Leakage check: remove training pairs that match eval =====
    eval_hashes = {hash_contract(str(c)) for c in eval_df["ground_truth_code"] if c}

    def check_leakage(df, name):
        before = len(df)
        df["_pos_hash"] = df["positive"].apply(lambda x: hash_contract(str(x)))
        df = df[~df["_pos_hash"].isin(eval_hashes)]
        df = df.drop(columns=["_pos_hash"])
        after = len(df)
        if before != after:
            log(f"  {name}: removed {before - after} pairs leaking into eval")
        return df

    real_df = check_leakage(real_df, "real")
    if len(synth_df) > 0:
        synth_df = check_leakage(synth_df, "synthetic")

    # ===== Merge =====
    if len(synth_df) > 0:
        # Ensure same columns
        for col in real_df.columns:
            if col not in synth_df.columns:
                synth_df[col] = ""
        for col in synth_df.columns:
            if col not in real_df.columns:
                real_df[col] = ""
        merged = pd.concat([real_df, synth_df[real_df.columns]], ignore_index=True)
    else:
        merged = real_df

    # Cross-source dedup by positive hash
    merged["_pos_hash"] = merged["positive"].apply(lambda x: hash_contract(str(x)))
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset="_pos_hash", keep="first")
    merged = merged.drop(columns=["_pos_hash"])
    log(f"After dedup: {len(merged)} (removed {before_dedup - len(merged)} dupes)")

    log(f"\nFINAL scar-pairs: {len(merged)}")
    log(f"  By source:")
    for src, count in merged["source"].value_counts().items():
        log(f"    {src}: {count}")

    # ===== Push scar-pairs =====
    log("\nPushing scar-pairs to HuggingFace...")
    pairs_ds = Dataset.from_pandas(merged)
    pairs_ds.push_to_hub(
        f"{HF_USERNAME}/scar-pairs",
        token=token,
        private=True,
    )
    log(f"Pushed {len(merged)} pairs to {HF_USERNAME}/scar-pairs")

    # ===== Push scar-corpus (copy from existing) =====
    log("\nCopying scar-corpus from existing scar-corpus...")
    corpus = load_dataset(f"{HF_USERNAME}/scar-corpus", split="train", token=token)
    corpus.push_to_hub(
        f"{HF_USERNAME}/scar-corpus",
        token=token,
        private=True,
    )
    log(f"Pushed {len(corpus)} contracts to {HF_USERNAME}/scar-corpus")

    return {
        "scar_pairs": len(merged),
        "scar_eval": len(eval_df),
        "scar_corpus": len(corpus),
        "real_pairs": len(real_df),
        "synthetic_pairs": len(synth_df),
    }


# ---------------------------------------------------------------------------
# Check mode: inspect all datasets
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[HF_SECRET],
    timeout=300,
)
def check_datasets():
    """Check stats for all SCAR datasets."""
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")

    for name in ["scar-corpus", "scar-pairs", "scar-eval"]:
        repo = f"{HF_USERNAME}/{name}"
        try:
            ds = load_dataset(repo, split="train", token=token)
            log(f"\n{name}: {len(ds)} rows")
            log(f"  columns: {ds.column_names}")
            if "source" in ds.column_names:
                import pandas as pd
                df = ds.to_pandas()
                log(f"  by source:")
                for src, count in df["source"].value_counts().items():
                    log(f"    {src}: {count}")
        except Exception as e:
            log(f"\n{name}: NOT FOUND — {e}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(mode: str = "eval_expand"):
    log(f"SCAR Data Pipeline (mode={mode})")

    if mode == "eval_expand":
        result = build_eval_set.remote()
        log(f"\nResult: {json.dumps(result, indent=2)}")

    elif mode == "synthetic":
        result = generate_synthetic_pairs.remote(
            n_contracts=5000,
            generations_per_contract=3,
            batch_size=50,
        )
        log(f"\nResult: {json.dumps(result, indent=2)}")

    elif mode == "merge_push":
        result = merge_and_push.remote()
        log(f"\nResult: {json.dumps(result, indent=2)}")

    elif mode == "check":
        check_datasets.remote()

    else:
        log(f"Unknown mode: {mode}. Use: eval_expand, synthetic, merge_push, check")

    log(f"\nDone (mode={mode})")
