"""
SCAR Solodit API Ingestion
===========================
Ingest 50k+ structured vulnerability findings from Cyfrin's Solodit API.
Each finding has: title, description, severity, code — exactly our contrastive pair format.

Modes:
  modal run scripts/scar_solodit_api.py --mode probe       # Test API, inspect response schema
  modal run scripts/scar_solodit_api.py --mode ingest      # Full ingestion (all 50k findings)
  modal run scripts/scar_solodit_api.py --mode build_pairs # Convert findings → contrastive pairs
  modal run scripts/scar_solodit_api.py --mode push        # Push to HF as scar-pairs

Requirements:
  - Modal secret: solodit-api-key (with SOLODIT_API_KEY=<your key>)
  - Modal secret: huggingface-token
"""

import modal
import os
import json
import hashlib
import time
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("scar-solodit-api")

HF_SECRET = modal.Secret.from_name("huggingface-token")
SOLODIT_SECRET = modal.Secret.from_name("solodit-api-key")
HF_USERNAME = "Farseen0"

VOLUME_NAME = "scar-data-vol"
DATA_DIR = "/scar_data"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "requests>=2.31.0",
        "datasets>=2.19.0",
        "huggingface_hub>=0.23.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
        "rank_bm25>=0.2.2",
    )
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# Solodit API client
# ---------------------------------------------------------------------------
API_BASE = "https://solodit.cyfrin.io/api/v1/solodit"
FINDINGS_ENDPOINT = f"{API_BASE}/findings"
RATE_LIMIT_DELAY = 4.0  # 20 req/60s → 4s between requests (conservative)


class SoloditClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "X-Cyfrin-API-Key": api_key,
            "Content-Type": "application/json",
        }
        self.last_request_time = 0

    def _rate_limit(self):
        """Respect 20 req/60s rate limit."""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def search_findings(self, page=1, page_size=100, severity=None,
                        keywords=None, tags=None, firms=None,
                        sort_field="Quality", sort_direction="Desc"):
        """Search findings with filters."""
        import requests

        self._rate_limit()

        body = {
            "page": page,
            "pageSize": page_size,
        }

        filters = {}
        if severity:
            filters["impact"] = severity if isinstance(severity, list) else [severity]
        if keywords:
            filters["keywords"] = keywords
        if tags:
            filters["tags"] = tags
        if firms:
            filters["firms"] = [{"value": f} for f in firms] if isinstance(firms, list) else [{"value": firms}]
        if sort_field:
            filters["sortField"] = sort_field
        if sort_direction:
            filters["sortDirection"] = sort_direction

        if filters:
            body["filters"] = filters

        resp = requests.post(FINDINGS_ENDPOINT, json=body, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_finding(self, slug: str):
        """Get full finding details by slug."""
        import requests

        self._rate_limit()

        url = f"{API_BASE}/finding/{slug}"
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Mode 1: Probe — test API and inspect response schema
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[SOLODIT_SECRET],
    timeout=120,
)
def probe_api():
    """Test API connectivity and inspect response schema."""
    api_key = os.environ.get("SOLODIT_API_KEY")
    if not api_key:
        raise ValueError("SOLODIT_API_KEY not set. Create Modal secret 'solodit-api-key'")

    client = SoloditClient(api_key)

    # Test 1: Basic search
    log("=== Test 1: Basic search (page 1, size 3) ===")
    result = client.search_findings(page=1, page_size=3)
    log(f"Response keys: {list(result.keys())}")
    log(f"Full response:\n{json.dumps(result, indent=2, default=str)[:5000]}")

    # If there are findings, inspect schema
    items_key = None
    for key in ["data", "findings", "items", "results"]:
        if key in result:
            items_key = key
            break

    if items_key:
        findings = result[items_key]
        if findings:
            log(f"\n=== Finding schema (first item) ===")
            first = findings[0]
            log(f"Keys: {list(first.keys())}")
            log(f"Full finding:\n{json.dumps(first, indent=2, default=str)[:3000]}")

    else:
        log(f"Couldn't find items key in response. Keys: {list(result.keys())}")

    # Test 3: Count total findings
    log("\n=== Test 3: Total counts ===")
    metadata = result.get("metadata", {})
    log(f"Metadata: {json.dumps(metadata, indent=2, default=str)[:2000]}")

    # Test 4: Check impact values across first 3 findings
    log("\n=== Test 4: Impact values ===")
    for item in (result.get("findings") or [])[:3]:
        log(f"  impact={item.get('impact')}, title={item.get('title', '')[:60]}")

    # Test 5: Check rateLimit info
    rate_limit = result.get("rateLimit", {})
    log(f"\n=== Test 5: Rate limit ===")
    log(f"Rate limit: {json.dumps(rate_limit, indent=2, default=str)}")

    # Test 6: Check tags structure
    log("\n=== Test 6: Tags structure ===")
    first_item = (result.get("findings") or [{}])[0]
    tags = first_item.get("issues_issuetagscore", [])
    log(f"Tags: {json.dumps(tags, indent=2, default=str)[:1000]}")
    log(f"Summary: {first_item.get('summary', '')[:500]}")

    return "Probe complete — check logs for schema"


# ---------------------------------------------------------------------------
# Mode 2: Full ingestion — paginate through all findings
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[SOLODIT_SECRET],
    volumes={DATA_DIR: vol},
    timeout=14400,  # 4 hours
    memory=16384,
)
def ingest_all_findings():
    """
    Paginate through all Solodit findings (50k+) and save to volume.
    Rate-limited to 20 req/60s → ~3.1s per page × 500 pages = ~26 min for 50k.
    """
    import pandas as pd

    api_key = os.environ.get("SOLODIT_API_KEY")
    client = SoloditClient(api_key)
    os.makedirs(DATA_DIR, exist_ok=True)

    all_findings = []
    page = 1
    page_size = 100
    total_expected = None

    # Resume from checkpoint if exists
    checkpoint_path = f"{DATA_DIR}/solodit_findings_checkpoint.json"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            all_findings = json.load(f)
        page = (len(all_findings) // page_size) + 1
        log(f"Resuming from checkpoint: {len(all_findings)} findings, starting page {page}")
    else:
        log("Starting full Solodit ingestion...")

    while True:
        try:
            result = client.search_findings(
                page=page,
                page_size=page_size,
                sort_field="Quality",
                sort_direction="Desc",
            )

            # Detect items key
            items = None
            for key in ["data", "findings", "items", "results"]:
                if key in result and isinstance(result[key], list):
                    items = result[key]
                    break

            if items is None:
                log(f"No items found in response at page {page}. Keys: {list(result.keys())}")
                break

            if total_expected is None:
                metadata = result.get("metadata", {})
                total_expected = metadata.get("totalResults") or metadata.get("total")
                log(f"Total findings: {total_expected}")

            all_findings.extend(items)

            if len(items) < page_size:
                log(f"Page {page}: got {len(items)} (last page). Total collected: {len(all_findings)}")
                break

            if page % 20 == 0:
                log(f"Page {page}: {len(all_findings)} findings so far...")

            # Checkpoint every 50 pages (5k findings)
            if page % 50 == 0:
                with open(checkpoint_path, "w") as f:
                    json.dump(all_findings, f)
                vol.commit()
                log(f"  Checkpoint saved: {len(all_findings)} findings")

            page += 1

        except Exception as e:
            log(f"Error at page {page}: {e}")
            if "429" in str(e) or "rate" in str(e).lower():
                log("Rate limited — waiting 60s...")
                time.sleep(60)
                continue
            else:
                log("Saving what we have and stopping...")
                break

    log(f"\nIngestion complete: {len(all_findings)} findings")

    # Save full dump
    output_path = f"{DATA_DIR}/solodit_findings_raw.json"
    with open(output_path, "w") as f:
        json.dump(all_findings, f)
    vol.commit()
    log(f"Saved to {output_path}")

    # Quick stats
    severities = {}
    for f in all_findings:
        sev = str(f.get("impact", "unknown")).upper()
        severities[sev] = severities.get(sev, 0) + 1
    log(f"By severity: {severities}")

    return {"total": len(all_findings), "severities": severities}


# ---------------------------------------------------------------------------
# Mode 3: Build contrastive pairs from ingested findings
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[SOLODIT_SECRET, HF_SECRET],
    volumes={DATA_DIR: vol},
    timeout=14400,
    memory=32768,
)
def build_pairs():
    """
    Convert raw Solodit findings → SCAR contrastive pairs.

    For each finding:
    - Query = "{SEVERITY} severity: {title}. {description[:300]}"
    - Positive = vulnerable code snippet
    - Hard negative = BM25-nearest code from different vulnerability

    Handles two cases:
    1. Finding has inline code snippets → extract from body/content
    2. Finding references external files → skip (no code to pair)
    """
    import pandas as pd
    from rank_bm25 import BM25Okapi
    from tqdm import tqdm

    api_key = os.environ.get("SOLODIT_API_KEY")
    token = os.environ.get("HF_TOKEN")
    vol.reload()

    # Load raw findings
    raw_path = f"{DATA_DIR}/solodit_findings_raw.json"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Run ingest mode first: {raw_path} not found")

    with open(raw_path, "r") as f:
        all_findings = json.load(f)
    log(f"Loaded {len(all_findings)} raw findings")

    # Load existing eval hashes (prevent leakage)
    eval_hashes = set()
    try:
        from datasets import load_dataset
        eval_ds = load_dataset(f"{HF_USERNAME}/scar-eval", split="train", token=token)
        for row in eval_ds:
            h = hashlib.sha256(
                re.sub(r'\s+', ' ', str(row.get("ground_truth_code", ""))).strip().encode()
            ).hexdigest()
            eval_hashes.add(h)
        log(f"Loaded {len(eval_hashes)} eval hashes for leakage prevention")
    except Exception as e:
        log(f"Could not load eval set: {e}")

    # Load existing training pair hashes (dedup)
    training_hashes = set()
    try:
        from datasets import load_dataset
        old_pairs = load_dataset(f"{HF_USERNAME}/scar-pairs", split="train", token=token)
        for row in old_pairs:
            h = hashlib.sha256(
                re.sub(r'\s+', ' ', str(row.get("positive", ""))).strip().encode()
            ).hexdigest()
            training_hashes.add(h)
        log(f"Loaded {len(training_hashes)} existing training hashes for dedup")
    except Exception as e:
        log(f"Could not load training pairs: {e}")

    def extract_code_from_content(content: str) -> list:
        """Extract Solidity code blocks from markdown content."""
        if not content:
            return []

        codes = []

        # Pattern 1: ```solidity ... ``` blocks
        solidity_blocks = re.findall(r'```(?:solidity|sol|javascript|js)?\s*\n(.*?)```',
                                      content, re.DOTALL)
        for block in solidity_blocks:
            block = block.strip()
            if len(block) >= 50 and any(kw in block for kw in
                ['function', 'contract', 'mapping', 'require', 'modifier',
                 'event', 'emit', 'msg.sender', 'address', 'uint']):
                codes.append(block)

        # Pattern 2: Indented code blocks (4+ spaces)
        if not codes:
            indented = re.findall(r'(?:^|\n)((?:    .+\n)+)', content)
            for block in indented:
                block = block.strip()
                if len(block) >= 50 and any(kw in block for kw in
                    ['function', 'contract', 'require', 'mapping']):
                    codes.append(block)

        return codes

    def normalize_severity(sev: str) -> str:
        """Normalize severity to CRITICAL/HIGH/MEDIUM."""
        s = str(sev).strip().upper()
        for valid in ("CRITICAL", "HIGH", "MEDIUM"):
            if valid in s:
                return "HIGH" if valid == "CRITICAL" else valid
        return ""

    # ===== Process findings into pairs =====
    pairs = []
    skipped_no_code = 0
    skipped_short = 0
    skipped_dup = 0
    skipped_leakage = 0

    log("Processing findings into contrastive pairs...")

    for finding in tqdm(all_findings):
        # Extract key fields (from probe: title, content, impact, firm_name, etc.)
        title = (finding.get("title") or "").strip()
        severity = normalize_severity(finding.get("impact") or "")
        if not severity or not title:
            continue

        # Get full content (markdown with code blocks)
        content = finding.get("content") or ""

        # Extract code snippets from content
        code_blocks = extract_code_from_content(content)
        if not code_blocks:
            skipped_no_code += 1
            continue

        # Use the largest code block as the "positive"
        positive_code = max(code_blocks, key=len)

        if len(positive_code) < 100:
            skipped_short += 1
            continue

        # Check for duplicates
        code_hash = hashlib.sha256(
            re.sub(r'\s+', ' ', positive_code).strip().encode()
        ).hexdigest()

        if code_hash in training_hashes:
            skipped_dup += 1
            continue

        if code_hash in eval_hashes:
            skipped_leakage += 1
            continue

        training_hashes.add(code_hash)

        # Build description — prefer summary field, fall back to stripped content
        summary = finding.get("summary") or ""
        if summary and len(summary.strip()) > 30:
            desc_text = re.sub(r'\s+', ' ', summary).strip()[:300]
        else:
            desc_text = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
            desc_text = re.sub(r'\s+', ' ', desc_text).strip()[:300]
        if len(desc_text) > 300:
            desc_text = desc_text[:300] + "..."

        query = f"{severity} severity: {title}. {desc_text}"

        # Extract metadata (from probe schema)
        tag_scores = finding.get("issues_issuetagscore", [])
        if isinstance(tag_scores, list):
            vuln_type = "|".join(
                str(t.get("tag", t.get("name", ""))) for t in tag_scores[:3]
                if isinstance(t, dict)
            )
        else:
            vuln_type = ""
        firm = finding.get("firm_name") or ""
        protocol = finding.get("protocol_name") or ""
        slug = finding.get("slug") or finding.get("id") or ""

        pairs.append({
            "query": query,
            "positive": positive_code,
            "hard_negative": "",  # Will be filled by BM25 mining
            "negative_type": "pending_bm25",
            "source": "Solodit-API",
            "severity": severity,
            "vuln_type": vuln_type,
            "quality_tier": 4,  # Highest tier — real audit findings
            "audit_firm": str(firm),
            "protocol_category": str(protocol),
            "solodit_slug": str(slug),
        })

    log(f"\n=== Pair extraction results ===")
    log(f"Total pairs: {len(pairs)}")
    log(f"Skipped (no code): {skipped_no_code}")
    log(f"Skipped (too short): {skipped_short}")
    log(f"Skipped (duplicate): {skipped_dup}")
    log(f"Skipped (eval leakage): {skipped_leakage}")

    if not pairs:
        log("No pairs extracted! Check finding schema in probe mode.")
        return {"pairs": 0}

    # ===== BM25 Hard Negative Mining =====
    log(f"\nMining BM25 hard negatives for {len(pairs)} pairs...")

    positives = [p["positive"] for p in pairs]
    tokenized = [re.sub(r'\s+', ' ', p).lower().split() for p in positives]

    bm25 = BM25Okapi(tokenized)

    for i, pair in enumerate(tqdm(pairs)):
        query_tokens = re.sub(r'\s+', ' ', pair["query"]).lower().split()
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)

        # Find nearest non-self match
        for j in ranked[:20]:
            if j != i and positives[j] != pair["positive"]:
                # Prefer different vuln type for hard negatives
                if pairs[j]["vuln_type"] != pair["vuln_type"] or j > ranked[5]:
                    pair["hard_negative"] = positives[j]
                    pair["negative_type"] = "bm25_mined"
                    break

    has_neg = sum(1 for p in pairs if p["negative_type"] == "bm25_mined")
    log(f"Hard negatives mined: {has_neg}/{len(pairs)}")

    # ===== Save =====
    df = pd.DataFrame(pairs)
    output_path = f"{DATA_DIR}/solodit_api_pairs.parquet"
    df.to_parquet(output_path, index=False)
    vol.commit()
    log(f"\nSaved {len(df)} pairs to {output_path}")

    # Stats
    log(f"\nBy severity:")
    for sev, count in df["severity"].value_counts().items():
        log(f"  {sev}: {count}")

    if "audit_firm" in df.columns:
        log(f"\nTop firms:")
        for firm, count in df["audit_firm"].value_counts().head(15).items():
            log(f"  {firm}: {count}")

    return {
        "total_pairs": len(df),
        "has_hard_neg": has_neg,
        "by_severity": df["severity"].value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# Mode 4: Push to HF — merge Solodit API pairs with existing training data
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes={DATA_DIR: vol},
    timeout=3600,
    memory=32768,
)
def push_merged():
    """Merge Solodit API pairs with existing training pairs, push to scar-pairs."""
    import pandas as pd
    from datasets import load_dataset, Dataset

    token = os.environ.get("HF_TOKEN")
    vol.reload()

    # Load Solodit API pairs
    solodit_path = f"{DATA_DIR}/solodit_api_pairs.parquet"
    if not os.path.exists(solodit_path):
        raise FileNotFoundError(f"Run build_pairs first: {solodit_path}")
    solodit_df = pd.read_parquet(solodit_path)
    log(f"Solodit API pairs: {len(solodit_df)}")

    # Load real holdout pairs (from eval_expand)
    real_path = f"{DATA_DIR}/scar_pairs_real.parquet"
    if os.path.exists(real_path):
        real_df = pd.read_parquet(real_path)
        log(f"Real holdout pairs: {len(real_df)}")
    else:
        # Fall back to original training data
        real_ds = load_dataset(f"{HF_USERNAME}/scar-pairs", split="train", token=token)
        real_df = real_ds.to_pandas()
        log(f"Original training pairs (no holdout applied): {len(real_df)}")

    # Align columns
    standard_cols = [
        "query", "positive", "hard_negative", "negative_type",
        "source", "severity", "vuln_type", "quality_tier",
    ]
    for col in standard_cols:
        if col not in solodit_df.columns:
            solodit_df[col] = ""
        if col not in real_df.columns:
            real_df[col] = ""

    # Merge
    merged = pd.concat([
        real_df[standard_cols],
        solodit_df[standard_cols],
    ], ignore_index=True)

    # Dedup by positive hash
    merged["_hash"] = merged["positive"].apply(
        lambda x: hashlib.sha256(re.sub(r'\s+', ' ', str(x)).strip().encode()).hexdigest()
    )
    before = len(merged)
    merged = merged.drop_duplicates(subset="_hash", keep="first")
    merged = merged.drop(columns=["_hash"])
    log(f"After dedup: {len(merged)} (removed {before - len(merged)} dupes)")

    # Leakage check against eval
    eval_path = f"{DATA_DIR}/scar_eval.parquet"
    if os.path.exists(eval_path):
        eval_df = pd.read_parquet(eval_path)
        eval_hashes = set()
        for code in eval_df["ground_truth_code"]:
            h = hashlib.sha256(re.sub(r'\s+', ' ', str(code)).strip().encode()).hexdigest()
            eval_hashes.add(h)

        merged["_hash"] = merged["positive"].apply(
            lambda x: hashlib.sha256(re.sub(r'\s+', ' ', str(x)).strip().encode()).hexdigest()
        )
        before = len(merged)
        merged = merged[~merged["_hash"].isin(eval_hashes)]
        merged = merged.drop(columns=["_hash"])
        log(f"After leakage removal: {len(merged)} (removed {before - len(merged)})")

    log(f"\n=== FINAL scar-pairs ===")
    log(f"Total: {len(merged)}")
    for src, count in merged["source"].value_counts().items():
        log(f"  {src}: {count}")

    # Push
    log("\nPushing to HuggingFace...")
    ds = Dataset.from_pandas(merged)
    ds.push_to_hub(f"{HF_USERNAME}/scar-pairs", token=token, private=True)
    log(f"Pushed {len(merged)} pairs to {HF_USERNAME}/scar-pairs")

    return {"total": len(merged), "sources": merged["source"].value_counts().to_dict()}


# ---------------------------------------------------------------------------
# Mode 5: Copy corpus from old to new SCAR-branded repo
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[HF_SECRET],
    timeout=7200,
    memory=32768,
)
def copy_corpus():
    """Copy scar-corpus → scar-corpus on HuggingFace."""
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")

    log("Loading scar-corpus (231k contracts)...")
    corpus = load_dataset(f"{HF_USERNAME}/scar-corpus", split="train", token=token)
    log(f"Loaded: {len(corpus)} contracts, columns: {corpus.column_names}")

    log("Pushing to scar-corpus...")
    corpus.push_to_hub(f"{HF_USERNAME}/scar-corpus", token=token, private=True)
    log(f"Done — pushed {len(corpus)} contracts to {HF_USERNAME}/scar-corpus")

    return {"contracts": len(corpus)}


# ---------------------------------------------------------------------------
# Mode 6: Audit — analyze quality of existing Solodit API pairs
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[HF_SECRET],
    timeout=600,
    memory=16384,
)
def audit_solodit_pairs():
    """Analyze quality issues in existing Solodit API pairs on HF."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(f"{HF_USERNAME}/scar-pairs", split="train", token=token)
    log(f"Total scar-pairs: {len(ds)}")

    # Filter to Solodit-API only
    solodit = [row for row in ds if row.get("source") == "Solodit-API"]
    log(f"Solodit-API pairs: {len(solodit)}")

    # Quality checks
    issues = {
        "short_code_under_5_lines": 0,
        "short_code_under_10_lines": 0,
        "no_function_body": 0,
        "is_interface_or_abstract": 0,
        "is_test_or_poc": 0,
        "is_javascript_not_solidity": 0,
        "short_query_under_80_chars": 0,
        "no_vulnerability_signal": 0,
    }

    # Vulnerability signal keywords (things that indicate actual vuln code)
    vuln_signals = [
        'msg.sender', 'msg.value', 'transfer(', 'call{', 'call(',
        'delegatecall(', 'selfdestruct(', 'tx.origin',
        'approve(', 'transferFrom(', 'balanceOf(',
        'abi.encode', 'keccak256(', 'ecrecover(',
        'storage ', 'memory ', 'unchecked {',
        'onlyOwner', 'require(', 'assert(', 'revert ',
        'external ', 'public ', 'internal ',
    ]

    examples = {"short": [], "no_func": [], "interface": [], "test": [], "js": []}

    for row in solodit:
        code = row.get("positive", "")
        query = row.get("query", "")
        lines = [l for l in code.split('\n') if l.strip() and not l.strip().startswith('//')]

        if len(lines) < 5:
            issues["short_code_under_5_lines"] += 1
            if len(examples["short"]) < 3:
                examples["short"].append(code[:200])
        if len(lines) < 10:
            issues["short_code_under_10_lines"] += 1

        if 'function ' not in code or ('{' not in code):
            issues["no_function_body"] += 1
            if len(examples["no_func"]) < 3:
                examples["no_func"].append(code[:200])

        if re.search(r'\b(interface|abstract contract)\b', code) and \
           not re.search(r'function\s+\w+[^;]+\{', code):
            issues["is_interface_or_abstract"] += 1
            if len(examples["interface"]) < 3:
                examples["interface"].append(code[:200])

        if any(kw in code for kw in ['setUp()', 'function test', 'forge-std', 'vm.prank', 'vm.expect']):
            issues["is_test_or_poc"] += 1
            if len(examples["test"]) < 3:
                examples["test"].append(code[:200])

        if any(kw in code for kw in ['console.log(', 'require(', 'module.exports']) and \
           'pragma solidity' not in code and 'contract ' not in code and 'function ' not in code:
            issues["is_javascript_not_solidity"] += 1
            if len(examples["js"]) < 3:
                examples["js"].append(code[:200])

        if len(query) < 80:
            issues["short_query_under_80_chars"] += 1

        if not any(sig in code for sig in vuln_signals):
            issues["no_vulnerability_signal"] += 1

    log(f"\n=== Quality Audit ({len(solodit)} Solodit-API pairs) ===")
    for issue, count in sorted(issues.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(solodit)
        log(f"  {issue}: {count} ({pct:.1f}%)")

    # Code length distribution
    code_lengths = [len(row.get("positive", "").split('\n')) for row in solodit]
    code_lengths.sort()
    log(f"\nCode length (lines) distribution:")
    for pct in [10, 25, 50, 75, 90]:
        idx = int(len(code_lengths) * pct / 100)
        log(f"  P{pct}: {code_lengths[idx]} lines")

    # Show examples of bad pairs
    log("\n=== Example issues ===")
    for category, exs in examples.items():
        if exs:
            log(f"\n--- {category} ---")
            for ex in exs[:2]:
                log(f"  {ex}")

    return {"total": len(solodit), "issues": issues}


# ---------------------------------------------------------------------------
# Mode 7: Rebuild — strict quality filters for Solodit API pairs
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[SOLODIT_SECRET, HF_SECRET],
    volumes={DATA_DIR: vol},
    timeout=14400,
    memory=32768,
)
def build_pairs_v2():
    """
    Rebuild Solodit API pairs with strict quality filters.

    Fixes from v1:
    1. No JS code blocks — Solidity only
    2. Min 8 non-empty non-comment lines (was 50 chars)
    3. Must contain function body with braces (not just declarations)
    4. No interfaces/abstract without implementations
    5. No test/PoC code (setUp, vm.prank, forge-std)
    6. Must have vulnerability-relevant patterns
    7. Description must be ≥50 meaningful chars
    8. Prefer code blocks with vulnerability signals over longest block
    """
    import pandas as pd
    from rank_bm25 import BM25Okapi
    from tqdm import tqdm

    api_key = os.environ.get("SOLODIT_API_KEY")
    token = os.environ.get("HF_TOKEN")
    vol.reload()

    # Load raw findings
    raw_path = f"{DATA_DIR}/solodit_findings_raw.json"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Run ingest mode first: {raw_path} not found")

    with open(raw_path, "r") as f:
        all_findings = json.load(f)
    log(f"Loaded {len(all_findings)} raw findings")

    # Load eval hashes (prevent leakage)
    eval_hashes = set()
    try:
        from datasets import load_dataset
        eval_ds = load_dataset(f"{HF_USERNAME}/scar-eval", split="train", token=token)
        for row in eval_ds:
            h = hashlib.sha256(
                re.sub(r'\s+', ' ', str(row.get("ground_truth_code", ""))).strip().encode()
            ).hexdigest()
            eval_hashes.add(h)
        log(f"Loaded {len(eval_hashes)} eval hashes for leakage prevention")
    except Exception as e:
        log(f"Could not load eval set: {e}")

    # Load existing non-Solodit training pair hashes (dedup)
    training_hashes = set()
    try:
        from datasets import load_dataset
        old_pairs = load_dataset(f"{HF_USERNAME}/scar-pairs", split="train", token=token)
        for row in old_pairs:
            if row.get("source") != "Solodit-API":
                h = hashlib.sha256(
                    re.sub(r'\s+', ' ', str(row.get("positive", ""))).strip().encode()
                ).hexdigest()
                training_hashes.add(h)
        log(f"Loaded {len(training_hashes)} non-Solodit training hashes for dedup")
    except Exception as e:
        log(f"Could not load training pairs: {e}")

    # --- Strict code extraction ---
    vuln_signals = [
        'msg.sender', 'msg.value', 'transfer(', 'call{', 'call(',
        'delegatecall(', 'selfdestruct(', 'tx.origin',
        'approve(', 'transferFrom(', 'balanceOf(',
        'abi.encode', 'keccak256(', 'ecrecover(',
        'unchecked {', 'onlyOwner', 'require(', 'assert(',
        'revert ', 'external ', 'public ', 'payable',
        '.slot', 'sload(', 'sstore(', 'assembly {',
    ]

    test_patterns = ['setUp()', 'function test', 'forge-std', 'vm.prank',
                     'vm.expect', 'vm.deal', 'vm.warp', 'assertEq(']

    def extract_code_strict(content: str) -> list:
        """Extract Solidity code blocks with strict quality filters."""
        if not content:
            return []

        # Only Solidity blocks — NO javascript/js
        blocks = re.findall(r'```(?:solidity|sol)?\s*\n(.*?)```', content, re.DOTALL)

        good_blocks = []
        for block in blocks:
            block = block.strip()

            # Count real lines (non-empty, non-comment)
            real_lines = [l for l in block.split('\n')
                          if l.strip() and not l.strip().startswith('//')]
            if len(real_lines) < 8:
                continue

            # Must contain function body (function + opening brace)
            if not re.search(r'function\s+\w+[^;]*\{', block, re.DOTALL):
                # Allow contracts with state variables + modifiers
                if not (re.search(r'contract\s+\w+', block) and '{' in block):
                    continue

            # No pure interfaces/abstract without implementations
            if re.search(r'\b(interface|abstract contract)\b', block):
                # Only keep if it also has function implementations
                if not re.search(r'function\s+\w+[^;]+\{[^}]', block, re.DOTALL):
                    continue

            # No test/PoC code
            if any(tp in block for tp in test_patterns):
                continue

            # Score by vulnerability signal density
            signal_count = sum(1 for sig in vuln_signals if sig in block)

            good_blocks.append((block, signal_count, len(real_lines)))

        return good_blocks

    def normalize_severity(sev: str) -> str:
        s = str(sev).strip().upper()
        for valid in ("CRITICAL", "HIGH", "MEDIUM"):
            if valid in s:
                return "HIGH" if valid == "CRITICAL" else valid
        return ""

    # ===== Process findings =====
    pairs = []
    stats = {
        "total_findings": len(all_findings),
        "no_severity_or_title": 0,
        "no_code_blocks": 0,
        "all_blocks_filtered": 0,
        "short_description": 0,
        "duplicate": 0,
        "eval_leakage": 0,
        "accepted": 0,
    }

    log("Processing findings with strict filters...")
    seen_hashes = set(training_hashes)

    for finding in tqdm(all_findings):
        title = (finding.get("title") or "").strip()
        severity = normalize_severity(finding.get("impact") or "")
        if not severity or not title:
            stats["no_severity_or_title"] += 1
            continue

        content = finding.get("content") or ""

        # Extract with strict filters
        scored_blocks = extract_code_strict(content)
        if not scored_blocks:
            # Check if original had code at all
            has_any_code = bool(re.findall(r'```', content))
            if has_any_code:
                stats["all_blocks_filtered"] += 1
            else:
                stats["no_code_blocks"] += 1
            continue

        # Pick block with highest vulnerability signal count, break ties by length
        scored_blocks.sort(key=lambda x: (x[1], x[2]), reverse=True)
        positive_code = scored_blocks[0][0]

        # Dedup
        code_hash = hashlib.sha256(
            re.sub(r'\s+', ' ', positive_code).strip().encode()
        ).hexdigest()

        if code_hash in seen_hashes:
            stats["duplicate"] += 1
            continue
        if code_hash in eval_hashes:
            stats["eval_leakage"] += 1
            continue
        seen_hashes.add(code_hash)

        # Build description — require ≥50 meaningful chars
        summary = finding.get("summary") or ""
        if summary and len(summary.strip()) > 50:
            desc_text = re.sub(r'\s+', ' ', summary).strip()[:300]
        else:
            desc_text = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
            desc_text = re.sub(r'\s+', ' ', desc_text).strip()[:300]

        if len(desc_text) < 50:
            stats["short_description"] += 1
            continue

        query = f"{severity} severity: {title}. {desc_text}"

        # Metadata
        tag_scores = finding.get("issues_issuetagscore", [])
        if isinstance(tag_scores, list):
            vuln_type = "|".join(
                str(t.get("tag", t.get("name", ""))) for t in tag_scores[:3]
                if isinstance(t, dict)
            )
        else:
            vuln_type = ""

        pairs.append({
            "query": query,
            "positive": positive_code,
            "hard_negative": "",
            "negative_type": "pending_bm25",
            "source": "Solodit-API",
            "severity": severity,
            "vuln_type": vuln_type,
            "quality_tier": 4,
        })
        stats["accepted"] += 1

    log(f"\n=== Strict filter results ===")
    for k, v in stats.items():
        log(f"  {k}: {v}")
    log(f"  acceptance_rate: {100*stats['accepted']/stats['total_findings']:.1f}%")

    if not pairs:
        log("No pairs passed filters!")
        return stats

    # ===== BM25 Hard Negative Mining =====
    log(f"\nMining BM25 hard negatives for {len(pairs)} pairs...")
    positives = [p["positive"] for p in pairs]
    tokenized = [re.sub(r'\s+', ' ', p).lower().split() for p in positives]
    bm25 = BM25Okapi(tokenized)

    for i, pair in enumerate(tqdm(pairs)):
        query_tokens = re.sub(r'\s+', ' ', pair["query"]).lower().split()
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)

        for j in ranked[:20]:
            if j != i and positives[j] != pair["positive"]:
                if pairs[j]["vuln_type"] != pair["vuln_type"] or j > ranked[5]:
                    pair["hard_negative"] = positives[j]
                    pair["negative_type"] = "bm25_mined"
                    break

    has_neg = sum(1 for p in pairs if p["negative_type"] == "bm25_mined")
    log(f"Hard negatives mined: {has_neg}/{len(pairs)}")

    # Save
    df = pd.DataFrame(pairs)
    output_path = f"{DATA_DIR}/solodit_api_pairs.parquet"
    df.to_parquet(output_path, index=False)
    vol.commit()
    log(f"\nSaved {len(df)} pairs to {output_path}")

    stats["final_pairs"] = len(df)
    stats["has_hard_neg"] = has_neg
    return stats


# ---------------------------------------------------------------------------
# Mode 8: Restructure pairs — 7k→scar-pairs, 17k→scar-pairs-extended
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[HF_SECRET],
    timeout=1200,
    memory=16384,
)
def restructure_pairs():
    """
    Restructure HF pair datasets:
    - Current scar-pairs (17,578) → scar-pairs-extended
    - scar-pairs (7,552) → scar-pairs (overwrite)
    """
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")

    # Step 1: Load current scar-pairs (17k) and push as scar-pairs-extended
    log("Loading scar-pairs (17,578 pairs)...")
    extended = load_dataset(f"{HF_USERNAME}/scar-pairs", split="train", token=token)
    log(f"Loaded: {len(extended)} pairs")

    log("Pushing to scar-pairs-extended...")
    extended.push_to_hub(f"{HF_USERNAME}/scar-pairs-extended", token=token, private=True)
    log(f"Pushed {len(extended)} pairs to scar-pairs-extended")

    # Step 2: Load scar-pairs (7k) and overwrite scar-pairs
    log("Loading scar-pairs (7,552 pairs)...")
    curated = load_dataset(f"{HF_USERNAME}/scar-pairs", split="train", token=token)
    log(f"Loaded: {len(curated)} pairs, columns: {curated.column_names}")

    log("Overwriting scar-pairs with 7,552 curated pairs...")
    curated.push_to_hub(f"{HF_USERNAME}/scar-pairs", token=token, private=True)
    log(f"Pushed {len(curated)} pairs to scar-pairs")

    return {
        "scar-pairs-extended": len(extended),
        "scar-pairs": len(curated),
    }


# ---------------------------------------------------------------------------
# Mode 9: Push dataset cards (README.md) to all SCAR HF datasets
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[HF_SECRET],
    timeout=300,
)
def push_dataset_cards():
    """Push README.md dataset cards to all SCAR HF dataset repos."""
    from huggingface_hub import HfApi
    import tempfile

    token = os.environ.get("HF_TOKEN")
    api = HfApi()

    cards = {
        f"{HF_USERNAME}/scar-pairs": """---
license: apache-2.0
task_categories:
  - text-retrieval
language:
  - en
tags:
  - smart-contracts
  - solidity
  - security
  - vulnerability-detection
  - contrastive-learning
size_categories:
  - 1K<n<10K
---

# SCAR Training Pairs

**7,552 human-auditor contrastive pairs** for training sparse retrieval models on smart contract vulnerability detection.

## Format

| Column | Description |
|--------|-------------|
| `query` | Audit finding: `"SEVERITY: [title]. [description]"` |
| `positive` | Vulnerable Solidity code snippet |
| `hard_negative` | Different vulnerability from same protocol |
| `source` | Dataset origin (Solodit, FORGE, DeFiHackLabs, etc.) |
| `severity` | HIGH / MEDIUM / LOW / CRITICAL |
| `vuln_type` | Vulnerability category |

## Sources

Solodit (2,060) | msc-audits-with-reasons (1,608) | msc-smart-contract-auditing (1,179) | DeFiHackLabs (608) | FORGE-Artifacts (532) | FORGE-Curated-v2 (323) | EVuLLM (288) | SmartBugs-Curated (128) | GitmateAI (22)

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Farseen0/scar-pairs", split="train")
```

## Related

- [scar-eval](https://huggingface.co/datasets/Farseen0/scar-eval) — 838 held-out eval pairs
- [scar-corpus](https://huggingface.co/datasets/Farseen0/scar-corpus) — 231k contract corpus
- [scar-weights](https://huggingface.co/Farseen0/scar-weights) — trained model weights
""",

        f"{HF_USERNAME}/scar-eval": """---
license: apache-2.0
task_categories:
  - text-retrieval
language:
  - en
tags:
  - smart-contracts
  - solidity
  - security
  - evaluation
  - benchmark
size_categories:
  - n<1K
---

# SCAR Evaluation Set

**838 held-out contrastive pairs** for evaluating smart contract vulnerability retrieval models. 10% stratified holdout from training sources plus 91 original FORGE-Curated pairs.

## Breakdown

| Source | Pairs |
|--------|-------|
| Solodit | 228 |
| msc-audits-with-reasons | 178 |
| msc-smart-contract-auditing | 131 |
| FORGE-Curated (original) | 91 |
| DeFiHackLabs | 67 |
| FORGE-Artifacts | 59 |
| FORGE-Curated-v2 | 36 |
| EVuLLM | 32 |
| SmartBugs-Curated | 14 |
| GitmateAI | 2 |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Farseen0/scar-eval", split="train")
```

## Related

- [scar-pairs](https://huggingface.co/datasets/Farseen0/scar-pairs) — 7,552 training pairs
- [scar-corpus](https://huggingface.co/datasets/Farseen0/scar-corpus) — 231k contract corpus
- [scar-weights](https://huggingface.co/Farseen0/scar-weights) — trained model weights
""",

        f"{HF_USERNAME}/scar-corpus": """---
license: apache-2.0
task_categories:
  - text-retrieval
language:
  - en
tags:
  - smart-contracts
  - solidity
  - security
  - code-corpus
size_categories:
  - 100K<n<1M
---

# SCAR Corpus

**231,269 Solidity smart contracts** for sparse retrieval over smart contract code. Used as the document collection for SCAR retrieval training and evaluation.

## Sources

- **DISL** — decompiled smart contracts (sampled 120k after dedup)
- **slither-audited** — contracts analyzed by Slither
- **DeFiVulnLabs** — 48 vulnerability types
- **FORGE-Curated** — curated audit contracts
- **SmartBugs Wild** — 47k real-world contracts

## Quality Filters

- Removed contracts < 50 lines
- Hash-based dedup on normalized source (strip comments + whitespace)
- Removed OpenZeppelin import-only files (>80% import statements)
- Cross-source dedup

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Farseen0/scar-corpus", split="train")
```

## Related

- [scar-pairs](https://huggingface.co/datasets/Farseen0/scar-pairs) — 7,552 training pairs
- [scar-eval](https://huggingface.co/datasets/Farseen0/scar-eval) — 838 eval pairs
- [scar-weights](https://huggingface.co/Farseen0/scar-weights) — trained model weights
""",

        f"{HF_USERNAME}/scar-pairs-extended": """---
license: apache-2.0
task_categories:
  - text-retrieval
language:
  - en
tags:
  - smart-contracts
  - solidity
  - security
  - contrastive-learning
size_categories:
  - 10K<n<100K
---

# SCAR Training Pairs (Extended)

**11,961 contrastive pairs** — an expanded version of [scar-pairs](https://huggingface.co/datasets/Farseen0/scar-pairs) that includes additional Solodit API-sourced pairs. The curated 7,552-pair subset (`scar-pairs`) was used for the final SCAR model; this extended set is provided for experimentation.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Farseen0/scar-pairs-extended", split="train")
```

## Related

- [scar-pairs](https://huggingface.co/datasets/Farseen0/scar-pairs) — 7,552 curated training pairs (used for final model)
- [scar-eval](https://huggingface.co/datasets/Farseen0/scar-eval) — 838 eval pairs
- [scar-corpus](https://huggingface.co/datasets/Farseen0/scar-corpus) — 231k contract corpus
""",
    }

    for repo_id, card_content in cards.items():
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(card_content)
            card_path = f.name

        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=repo_id, repo_type="dataset", token=token,
            commit_message="Add dataset card",
        )
        log(f"Pushed README.md to {repo_id}")

    return {"status": "done", "repos": list(cards.keys())}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(mode: str = "probe"):
    log(f"SCAR Solodit API Pipeline (mode={mode})")

    if mode == "probe":
        result = probe_api.remote()
        log(f"\n{result}")

    elif mode == "ingest":
        result = ingest_all_findings.remote()
        log(f"\nResult: {json.dumps(result, indent=2)}")

    elif mode == "build_pairs":
        result = build_pairs.remote()
        log(f"\nResult: {json.dumps(result, indent=2)}")

    elif mode == "push":
        result = push_merged.remote()
        log(f"\nResult: {json.dumps(result, indent=2)}")

    elif mode == "copy_corpus":
        result = copy_corpus.remote()
        log(f"\nResult: {json.dumps(result, indent=2)}")

    elif mode == "audit":
        result = audit_solodit_pairs.remote()
        log(f"\nResult: {json.dumps(result, indent=2)}")

    elif mode == "rebuild":
        log("Step 1: Rebuilding Solodit API pairs with strict filters...")
        result = build_pairs_v2.remote()
        log(f"\nResult: {json.dumps(result, indent=2)}")
        log("\nStep 2: Pushing merged dataset...")
        push_result = push_merged.remote()
        log(f"\nPush result: {json.dumps(push_result, indent=2)}")

    elif mode == "restructure_pairs":
        result = restructure_pairs.remote()
        log(f"\nResult: {json.dumps(result, indent=2)}")

    elif mode == "dataset_cards":
        result = push_dataset_cards.remote()
        log(f"\nResult: {json.dumps(result, indent=2)}")

    else:
        log(f"Unknown mode: {mode}. Use: probe, ingest, build_pairs, push, audit, rebuild, restructure_pairs")

    log(f"\nDone (mode={mode})")
