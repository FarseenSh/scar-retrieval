"""
SCAR Data Pipeline v2 — FORGE Extraction
=========================================
Extracts contrastive pairs from:
  A. FORGE-Curated  (flatten/vfp-vuln/ — pre-joined VFPs)
  B. FORGE-Artifacts (27k findings + 81k .sol files — needs joining)

Outputs pairs in the same format as pipeline.py Stream 2:
  {query, positive, hard_negative, negative_type, source, severity, vuln_type, quality_tier}

Usage:
  modal run scripts/data_pipeline_v2.py                    # both tracks
  modal run scripts/data_pipeline_v2.py --mode curated     # Track A only
  modal run scripts/data_pipeline_v2.py --mode artifacts   # Track B only
  modal run scripts/data_pipeline_v2.py --mode inspect     # inspect repos, no processing
  modal run scripts/data_pipeline_v2.py --mode merge       # merge with existing Stream 2 + push to HF
"""

import modal
import os

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("scar-data-pipeline-v2")

vol = modal.Volume.from_name("scar-intermediates", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "pandas>=2.0.0",
        "pyarrow>=15.0.0",
        "datasets>=2.19.0,<4.0.0",
        "huggingface_hub>=0.23.0",
        "tqdm>=4.66.0",
    )
)

INTERMEDIATE_DIR = "/intermediates"
HF_SECRET = modal.Secret.from_name("huggingface-token")
HF_USERNAME = "Farseen0"


def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
import re
import hashlib


def normalize_solidity(source: str) -> str:
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    source = re.sub(r'//.*$', '', source, flags=re.MULTILINE)
    source = re.sub(r'SPDX-License-Identifier:.*$', '', source, flags=re.MULTILINE)
    source = re.sub(r'pragma\s+solidity.*?;', '', source)
    lines = [line.strip() for line in source.splitlines()]
    lines = [line for line in lines if line]
    return '\n'.join(lines)


def hash_contract(source: str) -> str:
    normalized = normalize_solidity(source)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def build_query(severity: str, title: str, description: str) -> str:
    """Build query string in standard SCAR format."""
    sev = severity.upper().strip()
    if sev == 'CRITICAL':
        sev = 'HIGH'
    desc_trunc = description[:300].strip()
    if len(description) > 300:
        desc_trunc += '...'
    return f'{sev} severity: {title}. {desc_trunc}'


def extract_severity(raw) -> str:
    """Normalize severity from various formats."""
    if isinstance(raw, list):
        raw = raw[0] if raw else ''
    s = str(raw).strip().upper()
    # Normalize variants
    for valid in ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFORMATIONAL'):
        if valid in s:
            return valid
    return s


# ---------------------------------------------------------------------------
# Track A: FORGE-Curated (pre-joined VFPs)
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=3600)
def process_forge_curated():
    """Clone FORGE-Curated, extract pairs from flatten/vfp-vuln/ (pre-joined).

    The key fix vs pipeline.py: we DON'T need project_name→report_name matching.
    The VFP files already contain (findings + affected_files) pre-joined.
    We just iterate all VFPs and extract pairs directly.
    """
    import json
    import glob
    import random
    import subprocess
    import pandas as pd
    from collections import defaultdict

    random.seed(42)

    output_path = f"{INTERMEDIATE_DIR}/forge_curated_v2_pairs.parquet"
    vol.reload()
    if os.path.exists(output_path):
        df = pd.read_parquet(output_path)
        log(f"FORGE-Curated v2: already processed ({len(df)} pairs), skipping")
        return len(df)

    # Clone
    clone_dir = "/tmp/FORGE-Curated"
    if not os.path.exists(clone_dir):
        log("Cloning FORGE-Curated...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/shenyimings/FORGE-Curated.git", clone_dir],
            check=True, capture_output=True,
        )
    log("FORGE-Curated cloned")

    # Load ALL VFP files (not just vfp-vuln — we filter severity ourselves)
    vfp_vuln_dir = f'{clone_dir}/flatten/vfp-vuln'
    vfp_all_dir = f'{clone_dir}/flatten/vfp'

    # Prefer vfp-vuln (pre-filtered to medium+), fall back to vfp
    vfp_dir = vfp_vuln_dir if os.path.isdir(vfp_vuln_dir) else vfp_all_dir
    vfp_files = sorted(glob.glob(f'{vfp_dir}/*.json'))
    log(f"Found {len(vfp_files)} VFP files in {vfp_dir}")

    # Also load from vfp/ for any VFPs not in vfp-vuln/
    if os.path.isdir(vfp_all_dir) and vfp_dir != vfp_all_dir:
        all_files = sorted(glob.glob(f'{vfp_all_dir}/*.json'))
        vuln_basenames = {os.path.basename(f) for f in vfp_files}
        extra = [f for f in all_files if os.path.basename(f) not in vuln_basenames]
        if extra:
            log(f"Found {len(extra)} additional VFPs in {vfp_all_dir}")
            vfp_files.extend(extra)

    # Parse all VFPs
    vfps = []
    for vfp_path in vfp_files:
        try:
            with open(vfp_path, 'r', encoding='utf-8') as f:
                vfp = json.load(f)
            vfps.append(vfp)
        except Exception as e:
            log(f"  Skipping {os.path.basename(vfp_path)}: {e}")
            continue

    log(f"Loaded {len(vfps)} VFPs")

    # Extract pairs — NO report matching needed, VFPs are pre-joined
    all_entries = []
    stats = defaultdict(int)

    for vfp in vfps:
        affected_files = vfp.get('affected_files', {})
        if not affected_files:
            stats['no_affected_files'] += 1
            continue

        all_code = '\n'.join(affected_files.values())
        project_name = vfp.get('project_name', 'unknown')

        for finding in vfp.get('findings', []):
            severity = extract_severity(finding.get('severity', ''))

            if severity not in ('CRITICAL', 'HIGH', 'MEDIUM'):
                stats[f'skipped_{severity.lower()}'] += 1
                continue

            title = finding.get('title', '').strip()
            description = finding.get('description', '').strip()

            if not title or not description:
                stats['no_title_or_desc'] += 1
                continue

            # Match finding to specific files if possible
            finding_files = finding.get('files', [])
            if finding_files and isinstance(finding_files, list):
                matched_code = []
                for ff in finding_files:
                    fname = os.path.basename(str(ff))
                    if fname in affected_files:
                        matched_code.append(affected_files[fname])
                    else:
                        # Fuzzy match: try without path
                        for af_name, af_code in affected_files.items():
                            if fname.lower() == os.path.basename(af_name).lower():
                                matched_code.append(af_code)
                                break
                positive_code = '\n'.join(matched_code) if matched_code else all_code
            else:
                positive_code = all_code

            if not positive_code.strip() or len(positive_code.strip()) < 50:
                stats['code_too_short'] += 1
                continue

            # Extract CWE
            vuln_type = ''
            category = finding.get('category', {})
            if category and isinstance(category, dict):
                cwes = []
                for level_cwes in category.values():
                    if isinstance(level_cwes, list):
                        cwes.extend(level_cwes)
                vuln_type = '|'.join(cwes[:3])

            query = build_query(severity, title, description)
            all_entries.append({
                'query': query,
                'positive': positive_code,
                'severity': severity.upper() if severity.upper() != 'CRITICAL' else 'HIGH',
                'vuln_type': vuln_type,
                'title': title,
                'project_name': project_name,
            })
            stats['extracted'] += 1

    log(f"Extracted {len(all_entries)} findings. Stats: {dict(stats)}")

    # Build pairs with hard negatives (same-project different-finding)
    by_project = defaultdict(list)
    for entry in all_entries:
        by_project[entry['project_name']].append(entry)

    all_positives = [e['positive'] for e in all_entries]
    pairs = []

    for entry in all_entries:
        same_project = by_project[entry['project_name']]
        candidates = [e for e in same_project
                      if e['title'] != entry['title'] and e['positive'].strip()]

        if candidates:
            hard_neg = random.choice(candidates)['positive']
            neg_type = 'same_project_diff_finding'
        elif all_positives:
            hard_neg = random.choice(all_positives)
            neg_type = 'random'
        else:
            hard_neg = ''
            neg_type = 'none'

        pairs.append({
            'query': entry['query'],
            'positive': entry['positive'],
            'hard_negative': hard_neg,
            'negative_type': neg_type,
            'source': 'FORGE-Curated-v2',
            'severity': entry['severity'],
            'vuln_type': entry['vuln_type'],
            'quality_tier': 2,
        })

    # Dedup by (query_hash, positive_hash)
    seen = set()
    deduped = []
    for p in pairs:
        key = (hashlib.md5(p['query'].encode()).hexdigest(),
               hash_contract(p['positive']))
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    log(f"After dedup: {len(deduped)} pairs (removed {len(pairs) - len(deduped)} duplicates)")

    df = pd.DataFrame(deduped)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    df.to_parquet(output_path, index=False)
    vol.commit()

    log(f"FORGE-Curated v2: saved {len(df)} pairs to {output_path}")
    # Show severity distribution
    log(f"Severity distribution:\n{df['severity'].value_counts().to_string()}")
    log(f"Negative type distribution:\n{df['negative_type'].value_counts().to_string()}")
    log(f"Projects: {df['source'].value_counts().to_string()}")

    return len(df)


# ---------------------------------------------------------------------------
# Track B: FORGE-Artifacts (27k findings + 81k .sol files)
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=7200)
def process_forge_artifacts():
    """Clone FORGE-Artifacts, join findings→code, extract pairs.

    FORGE-Artifacts has:
    - dataset/results/*.json  — findings per report (27,497 total)
    - dataset/contracts/       — Solidity files organized by project

    We join via the 'files' and 'location' fields in findings.
    """
    import json
    import glob
    import random
    import subprocess
    import pandas as pd
    from collections import defaultdict
    from pathlib import Path

    random.seed(42)

    output_path = f"{INTERMEDIATE_DIR}/forge_artifacts_pairs.parquet"
    vol.reload()
    if os.path.exists(output_path):
        df = pd.read_parquet(output_path)
        log(f"FORGE-Artifacts: already processed ({len(df)} pairs), skipping")
        return len(df)

    # Clone (this is a large repo)
    clone_dir = "/tmp/FORGE-Artifacts"
    if not os.path.exists(clone_dir):
        log("Cloning FORGE-Artifacts (may take a few minutes)...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/shenyimings/FORGE-Artifacts.git", clone_dir],
            check=True, capture_output=True,
        )
    log("FORGE-Artifacts cloned")

    # ---- Step 1: Discover structure ----
    results_dir = f"{clone_dir}/dataset/results"
    contracts_dir = f"{clone_dir}/dataset/contracts"

    # Check if data exists or needs separate download
    result_files = sorted(glob.glob(f"{results_dir}/**/*.json", recursive=True))
    sol_files = sorted(glob.glob(f"{contracts_dir}/**/*.sol", recursive=True))

    log(f"Found {len(result_files)} result JSON files, {len(sol_files)} .sol files")

    if not result_files:
        # Data might be in a different location or need download
        # Check top-level structure
        all_dirs = []
        for d in Path(clone_dir).rglob('*'):
            if d.is_dir() and d.depth < 4:
                all_dirs.append(str(d))
        log(f"Top-level structure: {all_dirs[:30]}")

        # Try alternative paths
        for alt_results in [f"{clone_dir}/results", f"{clone_dir}/data/results"]:
            alt_files = glob.glob(f"{alt_results}/**/*.json", recursive=True)
            if alt_files:
                results_dir = alt_results
                result_files = sorted(alt_files)
                log(f"Found results in {alt_results}: {len(result_files)} files")
                break

        for alt_contracts in [f"{clone_dir}/contracts", f"{clone_dir}/data/contracts"]:
            alt_files = glob.glob(f"{alt_contracts}/**/*.sol", recursive=True)
            if alt_files:
                contracts_dir = alt_contracts
                sol_files = sorted(alt_files)
                log(f"Found contracts in {alt_contracts}: {len(sol_files)} files")
                break

    if not result_files:
        log("ERROR: No result JSON files found. Data may require separate download.")
        log("Checking README for download instructions...")
        readme_path = f"{clone_dir}/README.md"
        if os.path.exists(readme_path):
            with open(readme_path) as f:
                content = f.read()
            # Look for download instructions
            for line in content.split('\n'):
                if any(kw in line.lower() for kw in ['download', 'drive', 'r2', 'access', 'data']):
                    log(f"  README: {line.strip()}")
        return 0

    # ---- Step 2: Build contract index ----
    # Index all .sol files by various name keys for fuzzy matching
    log("Building contract index...")
    contract_index = {}  # various keys → file path
    contract_cache = {}  # file path → source code

    for sol_path in sol_files:
        rel_path = os.path.relpath(sol_path, contracts_dir)
        basename = os.path.basename(sol_path)

        # Index by multiple keys for fuzzy matching
        contract_index[rel_path.lower()] = sol_path
        contract_index[basename.lower()] = sol_path

        # Also index by path components (project/file.sol)
        parts = rel_path.split(os.sep)
        if len(parts) >= 2:
            short_path = f"{parts[-2]}/{parts[-1]}".lower()
            contract_index[short_path] = sol_path

    log(f"Contract index: {len(contract_index)} keys for {len(sol_files)} files")

    def load_contract(path):
        """Load and cache contract source."""
        if path not in contract_cache:
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    contract_cache[path] = f.read()
            except Exception:
                contract_cache[path] = ''
        return contract_cache[path]

    def find_contract(ref: str) -> str:
        """Find contract source by reference string (filename, path, or location)."""
        if not ref:
            return ''

        ref_str = str(ref).strip()

        # Parse location format: "Contract.sol::function#line-line"
        # or just "Contract.sol" or "path/to/Contract.sol"
        file_ref = ref_str.split('::')[0].split('#')[0].strip()

        # Try exact match
        if file_ref.lower() in contract_index:
            return load_contract(contract_index[file_ref.lower()])

        # Try basename
        basename = os.path.basename(file_ref).lower()
        if basename in contract_index:
            return load_contract(contract_index[basename])

        # Try fuzzy: strip common prefixes
        for prefix in ['contracts/', 'src/', 'lib/', './']:
            stripped = file_ref.lower().removeprefix(prefix)
            if stripped in contract_index:
                return load_contract(contract_index[stripped])

        return ''

    # ---- Step 3: Process findings ----
    log("Processing findings...")
    all_entries = []
    stats = defaultdict(int)

    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            stats['parse_error'] += 1
            continue

        # Handle both list and dict formats
        if isinstance(data, dict):
            # Could be a report with findings field
            findings = data.get('findings', [data])
            project_name = data.get('project_name', '') or data.get('project_info', {}).get('name', '')
            if not project_name:
                project_name = os.path.basename(os.path.dirname(result_file))
        elif isinstance(data, list):
            findings = data
            project_name = os.path.basename(os.path.dirname(result_file))
        else:
            stats['unknown_format'] += 1
            continue

        for finding in findings:
            if not isinstance(finding, dict):
                continue

            severity = extract_severity(finding.get('severity', ''))
            if severity not in ('CRITICAL', 'HIGH', 'MEDIUM'):
                stats[f'skip_{severity.lower() or "none"}'] += 1
                continue

            title = finding.get('title', '').strip()
            description = finding.get('description', '').strip()
            if not title or not description:
                stats['no_title_desc'] += 1
                continue

            # Find code via 'files', 'location', or 'affected_files'
            code_parts = []

            # Method 1: affected_files (if present, like FORGE-Curated)
            affected = finding.get('affected_files', {})
            if affected and isinstance(affected, dict):
                code_parts.extend(affected.values())

            # Method 2: files field
            if not code_parts:
                files_ref = finding.get('files', [])
                if isinstance(files_ref, str):
                    files_ref = [files_ref]
                if isinstance(files_ref, list):
                    for fref in files_ref:
                        code = find_contract(str(fref))
                        if code:
                            code_parts.append(code)

            # Method 3: location field
            if not code_parts:
                location = finding.get('location', '')
                if isinstance(location, list):
                    for loc in location:
                        code = find_contract(str(loc))
                        if code:
                            code_parts.append(code)
                elif location:
                    code = find_contract(str(location))
                    if code:
                        code_parts.append(code)

            if not code_parts:
                stats['no_code_found'] += 1
                continue

            positive_code = '\n'.join(code_parts)
            if len(positive_code.strip()) < 50:
                stats['code_too_short'] += 1
                continue

            # Extract CWE
            vuln_type = ''
            category = finding.get('category', {})
            if category and isinstance(category, dict):
                cwes = []
                for level_cwes in category.values():
                    if isinstance(level_cwes, list):
                        cwes.extend(level_cwes)
                vuln_type = '|'.join(cwes[:3])

            query = build_query(severity, title, description)
            all_entries.append({
                'query': query,
                'positive': positive_code,
                'severity': severity.upper() if severity.upper() != 'CRITICAL' else 'HIGH',
                'vuln_type': vuln_type,
                'title': title,
                'project_name': project_name,
            })
            stats['extracted'] += 1

    log(f"Extracted {len(all_entries)} findings from {len(result_files)} files")
    log(f"Stats: {dict(stats)}")

    if not all_entries:
        log("No pairs extracted. Check data format.")
        return 0

    # Build pairs with hard negatives
    by_project = defaultdict(list)
    for entry in all_entries:
        by_project[entry['project_name']].append(entry)

    all_positives = [e['positive'] for e in all_entries]
    pairs = []

    for entry in all_entries:
        same_project = by_project[entry['project_name']]
        candidates = [e for e in same_project
                      if e['title'] != entry['title'] and e['positive'].strip()]

        if candidates:
            hard_neg = random.choice(candidates)['positive']
            neg_type = 'same_project_diff_finding'
        elif all_positives:
            hard_neg = random.choice(all_positives)
            neg_type = 'random'
        else:
            hard_neg = ''
            neg_type = 'none'

        pairs.append({
            'query': entry['query'],
            'positive': entry['positive'],
            'hard_negative': hard_neg,
            'negative_type': neg_type,
            'source': 'FORGE-Artifacts',
            'severity': entry['severity'],
            'vuln_type': entry['vuln_type'],
            'quality_tier': 2,
        })

    # Dedup
    seen = set()
    deduped = []
    for p in pairs:
        key = (hashlib.md5(p['query'].encode()).hexdigest(),
               hash_contract(p['positive']))
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    log(f"After dedup: {len(deduped)} pairs (removed {len(pairs) - len(deduped)})")

    df = pd.DataFrame(deduped)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    df.to_parquet(output_path, index=False)
    vol.commit()

    log(f"FORGE-Artifacts: saved {len(df)} pairs to {output_path}")
    log(f"Severity: {df['severity'].value_counts().to_string()}")
    log(f"Neg types: {df['negative_type'].value_counts().to_string()}")
    log(f"Projects with pairs: {df['project_name'].nunique() if 'project_name' in df.columns else 'N/A'}")

    return len(df)


# ---------------------------------------------------------------------------
# Track C: EVuLLM + HF datasets (inspect + process)
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=3600)
def process_new_sources():
    """Inspect and process EVuLLM + 3 HF datasets.

    1. EVuLLM (CC0) — defihack.json: 389 entries with vulnerable_code_snippet
    2. GitmateAI/solidity_vulnerability_audit_dataset (Apache-2.0)
    3. SkywardNomad92/smart-contract-audit-findings (MIT)
    4. msc-smart-contract-auditing/audits-with-reasons (MIT)
    """
    import json
    import random
    import subprocess
    import pandas as pd
    from collections import defaultdict

    random.seed(42)
    vol.reload()

    output_path = f"{INTERMEDIATE_DIR}/new_sources_pairs.parquet"
    if os.path.exists(output_path):
        df = pd.read_parquet(output_path)
        log(f"New sources: already processed ({len(df)} pairs), skipping")
        return len(df)

    all_pairs = []

    # =============================================
    # 1. EVuLLM — defihack.json (CC0 license)
    # =============================================
    log("=" * 50)
    log("1. EVuLLM (CC0)")
    log("=" * 50)
    try:
        clone_dir = "/tmp/EVuLLM"
        if not os.path.exists(clone_dir):
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/Datalab-AUTH/EVuLLM-dataset.git", clone_dir],
                check=True, capture_output=True, timeout=120,
            )

        with open(f"{clone_dir}/defihack.json") as f:
            entries = json.load(f)

        log(f"  Loaded {len(entries)} entries from defihack.json")
        evullm_pairs = []
        for entry in entries:
            raw_code = entry.get('vulnerable_code_snippet', '')
            if isinstance(raw_code, list):
                code = '\n'.join(str(c) for c in raw_code).strip()
            else:
                code = str(raw_code).strip()
            title = str(entry.get('title', '')).strip()
            raw_rc = entry.get('root_cause', '')
            root_cause = str(raw_rc).strip() if not isinstance(raw_rc, list) else ', '.join(str(x) for x in raw_rc)
            raw_type = entry.get('type', '')
            vuln_type = str(raw_type).strip() if not isinstance(raw_type, list) else ', '.join(str(x) for x in raw_type)
            raw_analysis = entry.get('analysis', '')
            analysis = str(raw_analysis).strip() if not isinstance(raw_analysis, list) else ' '.join(str(x) for x in raw_analysis)

            if not code or not title or len(code) < 50:
                continue

            # Build query from available fields
            desc = analysis[:300] if analysis else root_cause
            if not desc:
                continue
            query = build_query('HIGH', f"{title} — {root_cause}", desc)

            evullm_pairs.append({
                'query': query,
                'positive': code,
                'hard_negative': '',  # filled later
                'negative_type': 'none',
                'source': 'EVuLLM',
                'severity': 'HIGH',
                'vuln_type': vuln_type,
                'quality_tier': 2,
                '_title': title,
            })

        # Add hard negatives (same vuln_type = hard, different = random)
        by_type = defaultdict(list)
        for p in evullm_pairs:
            by_type[p['vuln_type']].append(p)

        for p in evullm_pairs:
            same_type = [x for x in by_type[p['vuln_type']]
                         if x['_title'] != p['_title'] and x['positive'].strip()]
            if same_type:
                p['hard_negative'] = random.choice(same_type)['positive']
                p['negative_type'] = 'same_vuln_type'
            elif evullm_pairs:
                other = random.choice(evullm_pairs)
                p['hard_negative'] = other['positive']
                p['negative_type'] = 'random'

        # Clean up temp field
        for p in evullm_pairs:
            del p['_title']

        log(f"  EVuLLM: {len(evullm_pairs)} pairs extracted")
        all_pairs.extend(evullm_pairs)

    except Exception as e:
        log(f"  EVuLLM FAILED: {e}")

    # =============================================
    # 2. GitmateAI/solidity_vulnerability_audit_dataset (Apache-2.0)
    # =============================================
    log("\n" + "=" * 50)
    log("2. GitmateAI/solidity_vulnerability_audit_dataset")
    log("=" * 50)
    try:
        from datasets import load_dataset
        ds = load_dataset("GitmateAI/solidity_vulnerability_audit_dataset", trust_remote_code=True)
        for split in ds:
            log(f"  {split}: {len(ds[split])} rows, cols={ds[split].column_names}")
            # Show 2 samples
            for i in range(min(2, len(ds[split]))):
                row = ds[split][i]
                for k, v in row.items():
                    log(f"    {k}: {str(v)[:200]}")
                log("    ---")

        # Process if it has usable format
        gm_pairs = []
        for split in ds:
            for row in ds[split]:
                # Try to extract code and vulnerability description
                code = row.get('code', row.get('source_code', row.get('input', '')))
                vuln = row.get('vulnerability', row.get('output', row.get('label', '')))
                desc = row.get('description', row.get('explanation', ''))

                if not code or not vuln:
                    continue
                code = str(code).strip()
                vuln = str(vuln).strip()
                if len(code) < 50 or not vuln:
                    continue

                query = build_query('MEDIUM', vuln, desc if desc else vuln)
                gm_pairs.append({
                    'query': query,
                    'positive': code,
                    'hard_negative': '',
                    'negative_type': 'none',
                    'source': 'GitmateAI',
                    'severity': 'MEDIUM',
                    'vuln_type': vuln[:50],
                    'quality_tier': 3,
                })

        # Fill hard negatives
        if gm_pairs:
            for p in gm_pairs:
                other = random.choice(gm_pairs)
                p['hard_negative'] = other['positive']
                p['negative_type'] = 'random'

        log(f"  GitmateAI: {len(gm_pairs)} pairs extracted")
        all_pairs.extend(gm_pairs)

    except Exception as e:
        log(f"  GitmateAI FAILED: {e}")

    # =============================================
    # 3. SkywardNomad92/smart-contract-audit-findings (MIT)
    # =============================================
    log("\n" + "=" * 50)
    log("3. SkywardNomad92/smart-contract-audit-findings")
    log("=" * 50)
    try:
        ds = load_dataset("SkywardNomad92/smart-contract-audit-findings", trust_remote_code=True)
        for split in ds:
            log(f"  {split}: {len(ds[split])} rows, cols={ds[split].column_names}")
            for i in range(min(2, len(ds[split]))):
                row = ds[split][i]
                for k, v in row.items():
                    log(f"    {k}: {str(v)[:200]}")
                log("    ---")

        # Process
        sky_pairs = []
        for split in ds:
            for row in ds[split]:
                # Adapt based on actual columns
                code = ''
                query_text = ''
                severity = 'MEDIUM'

                # Try common column patterns
                for code_col in ['code', 'source_code', 'vulnerable_code', 'input', 'contract']:
                    if code_col in row and row[code_col]:
                        code = str(row[code_col]).strip()
                        break

                for desc_col in ['finding', 'description', 'vulnerability', 'output', 'title']:
                    if desc_col in row and row[desc_col]:
                        query_text = str(row[desc_col]).strip()
                        break

                for sev_col in ['severity', 'risk', 'impact']:
                    if sev_col in row and row[sev_col]:
                        severity = extract_severity(row[sev_col])
                        break

                if not code or not query_text or len(code) < 50:
                    continue
                if severity not in ('CRITICAL', 'HIGH', 'MEDIUM'):
                    continue

                query = build_query(severity, query_text[:100], query_text)
                sky_pairs.append({
                    'query': query,
                    'positive': code,
                    'hard_negative': '',
                    'negative_type': 'none',
                    'source': 'SkywardNomad92',
                    'severity': severity if severity != 'CRITICAL' else 'HIGH',
                    'vuln_type': '',
                    'quality_tier': 3,
                })

        if sky_pairs:
            for p in sky_pairs:
                other = random.choice(sky_pairs)
                p['hard_negative'] = other['positive']
                p['negative_type'] = 'random'

        log(f"  SkywardNomad92: {len(sky_pairs)} pairs extracted")
        all_pairs.extend(sky_pairs)

    except Exception as e:
        log(f"  SkywardNomad92 FAILED: {e}")

    # =============================================
    # 4. msc-smart-contract-auditing/audits-with-reasons (MIT)
    # =============================================
    log("\n" + "=" * 50)
    log("4. msc/audits-with-reasons")
    log("=" * 50)
    try:
        ds = load_dataset("msc-smart-contract-auditing/audits-with-reasons", trust_remote_code=True)
        for split in ds:
            log(f"  {split}: {len(ds[split])} rows, cols={ds[split].column_names}")
            for i in range(min(2, len(ds[split]))):
                row = ds[split][i]
                for k, v in row.items():
                    log(f"    {k}: {str(v)[:200]}")
                log("    ---")

        # Process
        msc2_pairs = []
        for split in ds:
            for row in ds[split]:
                code = ''
                query_text = ''
                severity = 'MEDIUM'

                for code_col in ['code', 'source_code', 'vulnerable_code', 'input', 'contract', 'source']:
                    if code_col in row and row[code_col]:
                        code = str(row[code_col]).strip()
                        break

                for desc_col in ['reason', 'finding', 'description', 'vulnerability', 'output', 'title']:
                    if desc_col in row and row[desc_col]:
                        query_text = str(row[desc_col]).strip()
                        break

                for sev_col in ['severity', 'risk', 'impact']:
                    if sev_col in row and row[sev_col]:
                        severity = extract_severity(row[sev_col])
                        break

                if not code or not query_text or len(code) < 50:
                    continue
                if severity not in ('CRITICAL', 'HIGH', 'MEDIUM'):
                    continue

                query = build_query(severity, query_text[:100], query_text)
                msc2_pairs.append({
                    'query': query,
                    'positive': code,
                    'hard_negative': '',
                    'negative_type': 'none',
                    'source': 'msc-audits-with-reasons',
                    'severity': severity if severity != 'CRITICAL' else 'HIGH',
                    'vuln_type': '',
                    'quality_tier': 3,
                })

        if msc2_pairs:
            for p in msc2_pairs:
                other = random.choice(msc2_pairs)
                p['hard_negative'] = other['positive']
                p['negative_type'] = 'random'

        log(f"  msc/audits-with-reasons: {len(msc2_pairs)} pairs extracted")
        all_pairs.extend(msc2_pairs)

    except Exception as e:
        log(f"  msc/audits-with-reasons FAILED: {e}")

    # =============================================
    # Combine, dedup, save
    # =============================================
    if not all_pairs:
        log("No pairs extracted from any new source")
        return 0

    # Dedup
    seen = set()
    deduped = []
    for p in all_pairs:
        key = (hashlib.md5(p['query'].encode()).hexdigest(),
               hash_contract(p['positive']))
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    log(f"\nTotal new source pairs: {len(deduped)} (from {len(all_pairs)} before dedup)")

    df = pd.DataFrame(deduped)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    df.to_parquet(output_path, index=False)
    vol.commit()

    log(f"Saved to {output_path}")
    log(f"Source distribution:\n{df['source'].value_counts().to_string()}")

    return len(df)


# ---------------------------------------------------------------------------
# Discover: check all candidate datasets from the discovery plan
# ---------------------------------------------------------------------------
@app.function(image=image, timeout=3600)
def discover_datasets():
    """Check all candidate datasets for usability, format, and license."""
    import json
    import glob
    import subprocess

    results = {}

    # =============================================
    # 1. DAppSCAN (GitHub: InPlusLab/DAppSCAN)
    # =============================================
    log("=" * 60)
    log("1. DAppSCAN (InPlusLab/DAppSCAN)")
    log("=" * 60)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/InPlusLab/DAppSCAN.git", "/tmp/DAppSCAN"],
            check=True, capture_output=True, timeout=300,
        )
        log("  Cloned successfully")

        # Structure
        swc_files = glob.glob("/tmp/DAppSCAN/**/SWC_source/**/*.json", recursive=True)
        sol_files = glob.glob("/tmp/DAppSCAN/**/*.sol", recursive=True)
        log(f"  SWC JSON files: {len(swc_files)}")
        log(f"  .sol files: {len(sol_files)}")

        # Top-level dirs
        top_dirs = sorted([d for d in os.listdir("/tmp/DAppSCAN") if os.path.isdir(f"/tmp/DAppSCAN/{d}")])
        log(f"  Top dirs: {top_dirs[:15]}")

        # Find any vulnerability data
        all_json = glob.glob("/tmp/DAppSCAN/**/*.json", recursive=True)
        log(f"  Total JSON files: {len(all_json)}")
        if all_json:
            for jf in all_json[:3]:
                log(f"  Sample JSON: {os.path.relpath(jf, '/tmp/DAppSCAN')}")
                try:
                    with open(jf) as f:
                        d = json.load(f)
                    if isinstance(d, list):
                        log(f"    List[{len(d)}], keys: {list(d[0].keys()) if d else 'empty'}")
                        if d:
                            log(f"    Entry[0]: {json.dumps(d[0], indent=2)[:400]}")
                    elif isinstance(d, dict):
                        log(f"    Dict keys: {list(d.keys())[:10]}")
                        log(f"    Preview: {json.dumps(d, indent=2)[:400]}")
                except Exception as e:
                    log(f"    Parse error: {e}")

        # License
        for lic in ["LICENSE", "LICENSE.md", "LICENSE.txt"]:
            lp = f"/tmp/DAppSCAN/{lic}"
            if os.path.exists(lp):
                with open(lp) as f:
                    log(f"  LICENSE: {f.read(300)}")
                break
        else:
            # Check README for license
            readme = "/tmp/DAppSCAN/README.md"
            if os.path.exists(readme):
                with open(readme) as f:
                    content = f.read()
                for line in content.split('\n'):
                    if 'license' in line.lower() or 'License' in line:
                        log(f"  README license: {line.strip()}")

        results['DAppSCAN'] = {'swc': len(swc_files), 'sol': len(sol_files), 'json': len(all_json)}
    except Exception as e:
        log(f"  FAILED: {e}")
        results['DAppSCAN'] = 'FAILED'

    # =============================================
    # 2. HuggingFace broad search
    # =============================================
    log("\n" + "=" * 60)
    log("2. HuggingFace Dataset Search")
    log("=" * 60)
    from huggingface_hub import list_datasets, dataset_info

    search_terms = [
        "smart contract vulnerability",
        "solidity vulnerability",
        "smart contract audit",
        "ethereum vulnerability",
        "smart contract security",
    ]

    seen = set()
    hf_results = []
    for term in search_terms:
        try:
            ds_list = list(list_datasets(search=term, sort="downloads", direction=-1))
            for d in ds_list[:10]:
                if d.id not in seen:
                    seen.add(d.id)
                    hf_results.append(d)
        except Exception:
            pass

    log(f"  Found {len(hf_results)} unique HF datasets")
    for d in sorted(hf_results, key=lambda x: x.downloads or 0, reverse=True)[:20]:
        # Check license
        lic = 'unknown'
        if hasattr(d, 'tags') and d.tags:
            lic_tags = [t for t in d.tags if t.startswith('license:')]
            if lic_tags:
                lic = lic_tags[0].replace('license:', '')
        log(f"  {d.id:55s} | dl={d.downloads or 0:>6} | lic={lic}")

    results['hf_search'] = len(hf_results)

    # =============================================
    # 3. SCV-1-2000 (darkknight25)
    # =============================================
    log("\n" + "=" * 60)
    log("3. SCV-1-2000 (darkknight25/Smart_Contract_Vulnerability_Dataset)")
    log("=" * 60)
    try:
        from datasets import load_dataset
        ds = load_dataset("darkknight25/Smart_Contract_Vulnerability_Dataset", trust_remote_code=True)
        for split in ds:
            log(f"  {split}: {len(ds[split])} rows")
            log(f"  Columns: {ds[split].column_names}")
            for i in range(min(2, len(ds[split]))):
                row = ds[split][i]
                log(f"  --- Example {i} ---")
                for k, v in row.items():
                    log(f"    {k}: {str(v)[:200]}")
        # License
        try:
            info = dataset_info("darkknight25/Smart_Contract_Vulnerability_Dataset")
            lic_tags = [t for t in (info.tags or []) if t.startswith('license:')]
            log(f"  License: {lic_tags if lic_tags else 'NOT SPECIFIED'}")
        except:
            pass
        results['SCV'] = {split: len(ds[split]) for split in ds}
    except Exception as e:
        log(f"  FAILED: {e}")
        results['SCV'] = 'FAILED'

    # =============================================
    # 4. Web3Bugs
    # =============================================
    log("\n" + "=" * 60)
    log("4. Web3Bugs")
    log("=" * 60)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ZhangZhuoSJTU/Web3Bugs.git", "/tmp/Web3Bugs"],
            check=True, capture_output=True, timeout=300,
        )
        log("  Cloned successfully")
        top = sorted(os.listdir("/tmp/Web3Bugs"))[:15]
        log(f"  Top-level: {top}")
        json_files = glob.glob("/tmp/Web3Bugs/**/*.json", recursive=True)
        md_files = glob.glob("/tmp/Web3Bugs/**/*.md", recursive=True)
        sol_files = glob.glob("/tmp/Web3Bugs/**/*.sol", recursive=True)
        log(f"  JSON: {len(json_files)}, MD: {len(md_files)}, SOL: {len(sol_files)}")

        # License
        for lic in ["LICENSE", "LICENSE.md"]:
            lp = f"/tmp/Web3Bugs/{lic}"
            if os.path.exists(lp):
                with open(lp) as f:
                    log(f"  LICENSE: {f.read(300)}")
                break
        else:
            log("  NO LICENSE FILE")

        # Sample
        if json_files:
            with open(json_files[0]) as f:
                log(f"  Sample JSON ({json_files[0]}): {f.read(400)}")

        results['Web3Bugs'] = {'json': len(json_files), 'md': len(md_files), 'sol': len(sol_files)}
    except Exception as e:
        log(f"  FAILED: {e}")
        results['Web3Bugs'] = 'FAILED'

    # =============================================
    # 5. EVuLLM Dataset
    # =============================================
    log("\n" + "=" * 60)
    log("5. EVuLLM Dataset (Datalab-AUTH/EVuLLM-dataset)")
    log("=" * 60)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/Datalab-AUTH/EVuLLM-dataset.git", "/tmp/EVuLLM"],
            check=True, capture_output=True, timeout=120,
        )
        log("  Cloned successfully")
        top = sorted(os.listdir("/tmp/EVuLLM"))[:15]
        log(f"  Top-level: {top}")
        all_files = glob.glob("/tmp/EVuLLM/**/*.*", recursive=True)
        by_ext = {}
        for f in all_files:
            ext = os.path.splitext(f)[1]
            by_ext[ext] = by_ext.get(ext, 0) + 1
        log(f"  File types: {by_ext}")

        # Show data files
        data_files = [f for f in all_files if f.endswith(('.json', '.jsonl', '.csv', '.parquet'))]
        for df in data_files[:5]:
            size = os.path.getsize(df)
            log(f"  Data file: {os.path.relpath(df, '/tmp/EVuLLM')} ({size:,} bytes)")
            try:
                if df.endswith('.json'):
                    with open(df) as fh:
                        d = json.load(fh)
                    if isinstance(d, list):
                        log(f"    List[{len(d)}]")
                        if d:
                            log(f"    Keys: {list(d[0].keys()) if isinstance(d[0], dict) else type(d[0])}")
                            log(f"    Sample: {json.dumps(d[0], indent=2)[:400]}")
                    elif isinstance(d, dict):
                        log(f"    Dict keys: {list(d.keys())[:10]}")
                elif df.endswith('.jsonl'):
                    with open(df) as fh:
                        lines = [fh.readline() for _ in range(3)]
                    for l in lines:
                        if l.strip():
                            log(f"    JSONL sample: {l.strip()[:400]}")
                elif df.endswith('.csv'):
                    import pandas as pd
                    ddf = pd.read_csv(df, nrows=3)
                    log(f"    CSV columns: {list(ddf.columns)}")
                    log(f"    Rows: {len(pd.read_csv(df))}")
            except Exception as e:
                log(f"    Parse error: {e}")

        # License
        for lic in ["LICENSE", "LICENSE.md"]:
            lp = f"/tmp/EVuLLM/{lic}"
            if os.path.exists(lp):
                with open(lp) as f:
                    log(f"  LICENSE: {f.read(300)}")
                break
        else:
            log("  NO LICENSE FILE")

        results['EVuLLM'] = by_ext
    except Exception as e:
        log(f"  FAILED: {e}")
        results['EVuLLM'] = 'FAILED'

    # =============================================
    # 6. ReentrancyStudy-Data
    # =============================================
    log("\n" + "=" * 60)
    log("6. ReentrancyStudy-Data")
    log("=" * 60)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/InPlusLab/ReentrancyStudy-Data.git", "/tmp/ReentrancyStudy"],
            check=True, capture_output=True, timeout=120,
        )
        log("  Cloned successfully")
        sol_files = glob.glob("/tmp/ReentrancyStudy/**/*.sol", recursive=True)
        json_files = glob.glob("/tmp/ReentrancyStudy/**/*.json", recursive=True)
        csv_files = glob.glob("/tmp/ReentrancyStudy/**/*.csv", recursive=True)
        log(f"  .sol: {len(sol_files)}, .json: {len(json_files)}, .csv: {len(csv_files)}")

        top = sorted(os.listdir("/tmp/ReentrancyStudy"))[:15]
        log(f"  Top-level: {top}")

        # Sample data
        if csv_files:
            import pandas as pd
            for cf in csv_files[:2]:
                df = pd.read_csv(cf, nrows=3)
                log(f"  CSV {os.path.basename(cf)}: cols={list(df.columns)}, rows={len(pd.read_csv(cf))}")
        if json_files:
            with open(json_files[0]) as f:
                log(f"  JSON sample: {f.read(400)}")

        # License
        for lic in ["LICENSE", "LICENSE.md"]:
            lp = f"/tmp/ReentrancyStudy/{lic}"
            if os.path.exists(lp):
                with open(lp) as f:
                    log(f"  LICENSE: {f.read(300)}")
                break
        else:
            log("  NO LICENSE FILE")

        results['ReentrancyStudy'] = {'sol': len(sol_files), 'json': len(json_files), 'csv': len(csv_files)}
    except Exception as e:
        log(f"  FAILED: {e}")
        results['ReentrancyStudy'] = 'FAILED'

    # =============================================
    # 7. MetaTrustSig / iAudit (HF + GitHub)
    # =============================================
    log("\n" + "=" * 60)
    log("7. MetaTrustSig / iAudit / TrustLLM")
    log("=" * 60)
    from huggingface_hub import list_repo_tree

    for repo in ["MetaTrustSig/TrustLLM", "MetaTrustSig/iAudit",
                  "MetaTrustSig/iAudit-dataset"]:
        try:
            files = list(list_repo_tree(repo, repo_type="dataset"))
            log(f"  HF dataset {repo}: {len(files)} files")
            for f in files[:10]:
                log(f"    {f.path} ({getattr(f, 'size', 'dir')})")
        except Exception:
            pass
        try:
            files = list(list_repo_tree(repo, repo_type="model"))
            log(f"  HF model {repo}: {len(files)} files")
            for f in files[:10]:
                log(f"    {f.path} ({getattr(f, 'size', 'dir')})")
        except Exception:
            pass

    # Try GitHub
    for gh_repo in ["AuditWen/iAudit", "AuditWen/iAudit-dataset",
                     "AuditWen/iAudit-Data"]:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 f"https://github.com/{gh_repo}.git", f"/tmp/{gh_repo.split('/')[-1]}"],
                check=True, capture_output=True, timeout=60,
            )
            log(f"  GitHub {gh_repo}: cloned")
            top = sorted(os.listdir(f"/tmp/{gh_repo.split('/')[-1]}"))[:10]
            log(f"    Files: {top}")
        except Exception:
            pass

    # Also check the iAudit paper's GitHub (ICSE 2025)
    for gh_repo in ["AuditWen/iAudit-ICSE2025",
                     "AuditWen/iAudit"]:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 f"https://github.com/{gh_repo}.git", f"/tmp/iAudit_check"],
                check=True, capture_output=True, timeout=60,
            )
            log(f"  GitHub {gh_repo}: cloned")
            all_f = glob.glob("/tmp/iAudit_check/**/*.*", recursive=True)
            by_ext = {}
            for f in all_f:
                ext = os.path.splitext(f)[1]
                by_ext[ext] = by_ext.get(ext, 0) + 1
            log(f"    File types: {by_ext}")
            break
        except Exception:
            pass

    # =============================================
    # 8. Additional HF datasets worth checking
    # =============================================
    log("\n" + "=" * 60)
    log("8. Specific HF datasets")
    log("=" * 60)
    specific_hf = [
        "forta/malicious-smart-contract-dataset",
        "peterxyz/smart-contract-vuln-detection",
    ]
    from datasets import load_dataset
    for ds_name in specific_hf:
        try:
            ds = load_dataset(ds_name, trust_remote_code=True)
            log(f"  {ds_name}:")
            for split in ds:
                log(f"    {split}: {len(ds[split])} rows, cols={ds[split].column_names}")
                if len(ds[split]) > 0:
                    row = ds[split][0]
                    for k, v in row.items():
                        log(f"      {k}: {str(v)[:200]}")
            # License
            try:
                info = dataset_info(ds_name)
                lic_tags = [t for t in (info.tags or []) if t.startswith('license:')]
                log(f"    License: {lic_tags}")
            except:
                pass
        except Exception as e:
            log(f"  {ds_name}: FAILED ({e})")

    # =============================================
    # SUMMARY
    # =============================================
    log("\n" + "=" * 60)
    log("DISCOVERY SUMMARY")
    log("=" * 60)
    log(json.dumps(results, indent=2, default=str))

    return results


# ---------------------------------------------------------------------------
# Inspect mode: check repo structure without processing
# ---------------------------------------------------------------------------
@app.function(image=image, timeout=1800)
def inspect_repos():
    """Clone both repos and report structure + sample data."""
    import json
    import glob
    import subprocess

    results = {}

    # --- FORGE-Curated ---
    log("=== FORGE-Curated ===")
    clone_dir = "/tmp/FORGE-Curated"
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/shenyimings/FORGE-Curated.git", clone_dir],
        check=True, capture_output=True,
    )

    vfp_vuln = sorted(glob.glob(f"{clone_dir}/flatten/vfp-vuln/*.json"))
    vfp_all = sorted(glob.glob(f"{clone_dir}/flatten/vfp/*.json"))
    log(f"  VFP-vuln files: {len(vfp_vuln)}")
    log(f"  VFP-all files:  {len(vfp_all)}")

    # Show a sample
    if vfp_vuln:
        with open(vfp_vuln[0]) as f:
            sample = json.load(f)
        log(f"  Sample VFP keys: {list(sample.keys())}")
        log(f"  project_name: {sample.get('project_name', 'N/A')}")
        log(f"  # findings: {len(sample.get('findings', []))}")
        log(f"  # affected_files: {len(sample.get('affected_files', {}))}")
        if sample.get('findings'):
            f0 = sample['findings'][0]
            log(f"  Finding[0] keys: {list(f0.keys())}")
            log(f"  Finding[0] title: {f0.get('title', 'N/A')}")
            log(f"  Finding[0] severity: {f0.get('severity', 'N/A')}")
            log(f"  Finding[0] description[:200]: {str(f0.get('description', ''))[:200]}")
        if sample.get('affected_files'):
            for fname, code in list(sample['affected_files'].items())[:1]:
                log(f"  affected_files[{fname}]: {len(code)} chars")
                log(f"    code[:200]: {code[:200]}")

    results['curated_vfp_vuln'] = len(vfp_vuln)
    results['curated_vfp_all'] = len(vfp_all)

    # Check license
    lic_path = f"{clone_dir}/LICENSE"
    if os.path.exists(lic_path):
        with open(lic_path) as f:
            log(f"  LICENSE: {f.read(200)}")

    # --- FORGE-Artifacts ---
    log("\n=== FORGE-Artifacts ===")
    clone_dir2 = "/tmp/FORGE-Artifacts"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/shenyimings/FORGE-Artifacts.git", clone_dir2],
            check=True, capture_output=True, timeout=600,
        )
        log("  Cloned successfully")
    except Exception as e:
        log(f"  Clone failed: {e}")
        log("  Checking if data needs separate download...")
        results['artifacts_clone'] = 'FAILED'
        return results

    # Check structure
    result_files = sorted(glob.glob(f"{clone_dir2}/dataset/results/**/*.json", recursive=True))
    sol_files = sorted(glob.glob(f"{clone_dir2}/dataset/contracts/**/*.sol", recursive=True))
    log(f"  Result JSONs: {len(result_files)}")
    log(f"  .sol files: {len(sol_files)}")

    # Check alternative locations
    if not result_files:
        log("  No results in dataset/results/, scanning...")
        all_json = glob.glob(f"{clone_dir2}/**/*.json", recursive=True)
        log(f"  Total .json files anywhere: {len(all_json)}")
        for j in all_json[:10]:
            log(f"    {os.path.relpath(j, clone_dir2)}")

    if not sol_files:
        log("  No .sol in dataset/contracts/, scanning...")
        all_sol = glob.glob(f"{clone_dir2}/**/*.sol", recursive=True)
        log(f"  Total .sol files anywhere: {len(all_sol)}")

    # Show sample result
    if result_files:
        with open(result_files[0]) as f:
            sample = json.load(f)
        if isinstance(sample, list):
            log(f"  Sample result: list of {len(sample)} entries")
            if sample:
                log(f"  Entry[0] keys: {list(sample[0].keys())}")
                log(f"  Entry[0]: {json.dumps(sample[0], indent=2)[:500]}")
        elif isinstance(sample, dict):
            log(f"  Sample result keys: {list(sample.keys())}")
            log(f"  Sample: {json.dumps(sample, indent=2)[:500]}")

    # Check README
    readme = f"{clone_dir2}/README.md"
    if os.path.exists(readme):
        with open(readme) as f:
            content = f.read()
        for line in content.split('\n'):
            if any(kw in line.lower() for kw in ['download', 'drive', 'data', 'access', 'r2', 'setup']):
                log(f"  README: {line.strip()}")

    # Check license
    lic_path = f"{clone_dir2}/LICENSE"
    if os.path.exists(lic_path):
        with open(lic_path) as f:
            log(f"  LICENSE: {f.read(200)}")

    results['artifacts_results'] = len(result_files)
    results['artifacts_sols'] = len(sol_files)

    return results


# ---------------------------------------------------------------------------
# Merge: combine new FORGE pairs with existing Stream 2 + push to HF
# ---------------------------------------------------------------------------
@app.function(
    image=image, volumes={INTERMEDIATE_DIR: vol},
    secrets=[HF_SECRET], timeout=3600,
)
def merge_and_push():
    """Merge new FORGE pairs with existing Stream 2, dedup, push to HF."""
    import pandas as pd
    from datasets import Dataset

    vol.reload()

    # Load existing Stream 2
    existing_path = f"{INTERMEDIATE_DIR}/stream2_merged.parquet"
    if os.path.exists(existing_path):
        existing = pd.read_parquet(existing_path)
        log(f"Existing Stream 2: {len(existing)} pairs")
        log(f"  Sources: {existing['source'].value_counts().to_string()}")
    else:
        log("WARNING: No existing stream2_merged.parquet found")
        existing = pd.DataFrame()

    # Load new FORGE pairs
    new_dfs = []

    curated_path = f"{INTERMEDIATE_DIR}/forge_curated_v2_pairs.parquet"
    if os.path.exists(curated_path):
        curated = pd.read_parquet(curated_path)
        log(f"FORGE-Curated v2: {len(curated)} new pairs")
        new_dfs.append(curated)

    artifacts_path = f"{INTERMEDIATE_DIR}/forge_artifacts_pairs.parquet"
    if os.path.exists(artifacts_path):
        artifacts = pd.read_parquet(artifacts_path)
        log(f"FORGE-Artifacts: {len(artifacts)} new pairs")
        new_dfs.append(artifacts)

    new_sources_path = f"{INTERMEDIATE_DIR}/new_sources_pairs.parquet"
    if os.path.exists(new_sources_path):
        new_src = pd.read_parquet(new_sources_path)
        log(f"New sources (EVuLLM+HF): {len(new_src)} new pairs")
        new_dfs.append(new_src)

    if not new_dfs:
        log("No new FORGE pairs found. Run curated/artifacts mode first.")
        return

    new_pairs = pd.concat(new_dfs, ignore_index=True)

    # Remove old FORGE-Curated pairs from existing (we're replacing them)
    if len(existing) > 0 and 'source' in existing.columns:
        old_forge = (existing['source'] == 'FORGE-Curated').sum()
        existing = existing[existing['source'] != 'FORGE-Curated'].reset_index(drop=True)
        log(f"Removed {old_forge} old FORGE-Curated pairs from existing")

    # Combine
    if len(existing) > 0:
        merged = pd.concat([existing, new_pairs], ignore_index=True)
    else:
        merged = new_pairs

    # Cross-dedup by positive hash
    merged['_pos_hash'] = merged['positive'].apply(hash_contract)
    before = len(merged)
    merged = merged.drop_duplicates(subset='_pos_hash', keep='first')
    merged = merged.drop(columns='_pos_hash').reset_index(drop=True)
    log(f"After cross-source dedup: {before} → {len(merged)}")

    # Leakage check against Stream 3 eval
    stream3_path = f"{INTERMEDIATE_DIR}/forge_stream3_eval.parquet"
    if os.path.exists(stream3_path):
        s3 = pd.read_parquet(stream3_path)
        s3_hashes = set()
        for code in s3.get('positive', s3.get('code', [])):
            if isinstance(code, str) and code.strip():
                s3_hashes.add(hash_contract(code))
        leaked = []
        for idx, row in merged.iterrows():
            if hash_contract(row['positive']) in s3_hashes:
                leaked.append(idx)
        if leaked:
            log(f"LEAKAGE: removing {len(leaked)} pairs that overlap with Stream 3 eval")
            merged = merged.drop(leaked).reset_index(drop=True)

    # Save locally
    output_path = f"{INTERMEDIATE_DIR}/stream2_merged_v2.parquet"
    merged.to_parquet(output_path, index=False)
    vol.commit()

    log(f"\nFinal merged dataset: {len(merged)} pairs")
    log(f"Source distribution:\n{merged['source'].value_counts().to_string()}")
    log(f"Severity distribution:\n{merged['severity'].value_counts().to_string()}")

    # Push to HF
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_ds = Dataset.from_pandas(merged, preserve_index=False)
        repo_id = f"{HF_USERNAME}/scar-pairs"
        hf_ds.push_to_hub(
            repo_id, private=True, token=hf_token,
            commit_message=f'v2: {len(merged)} pairs (added FORGE-Artifacts + fixed FORGE-Curated)',
        )
        log(f"Pushed to {repo_id}")
    else:
        log("WARNING: No HF_TOKEN, skipping push")

    return len(merged)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(mode: str = "both"):
    """
    Modes:
      inspect   — clone repos, show structure, no processing
      curated   — Track A: FORGE-Curated VFPs only
      artifacts — Track B: FORGE-Artifacts joining only
      both      — Track A + B in parallel
      merge     — merge new pairs with existing Stream 2, push to HF
      all       — both + merge
    """
    import json
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] SCAR Data Pipeline v2 (mode={mode})")

    if mode == "inspect":
        print("--- INSPECTING FORGE REPOS ---")
        result = inspect_repos.remote()
        print(f"\nInspection results: {result}")
        return

    if mode == "discover":
        print("--- DATASET DISCOVERY (checking all candidates) ---")
        result = discover_datasets.remote()
        print(f"\nDiscovery results: {json.dumps(result, indent=2, default=str)}")
        return

    if mode in ("curated", "both", "all"):
        print("--- Track A: FORGE-Curated ---")
        if mode in ("both", "all"):
            h_curated = process_forge_curated.spawn()
        else:
            n = process_forge_curated.remote()
            print(f"  FORGE-Curated: {n} pairs")

    if mode in ("artifacts", "both", "all"):
        print("--- Track B: FORGE-Artifacts ---")
        if mode in ("both", "all"):
            h_artifacts = process_forge_artifacts.spawn()
        else:
            n = process_forge_artifacts.remote()
            print(f"  FORGE-Artifacts: {n} pairs")

    if mode in ("new_sources", "all"):
        print("--- Track C: EVuLLM + HF datasets ---")
        if mode == "all":
            h_new = process_new_sources.spawn()
        else:
            n = process_new_sources.remote()
            print(f"  New sources: {n} pairs")

    # Wait for parallel tasks
    if mode in ("both", "all"):
        n_curated = h_curated.get()
        print(f"  FORGE-Curated: {n_curated} pairs")
        n_artifacts = h_artifacts.get()
        print(f"  FORGE-Artifacts: {n_artifacts} pairs")

    if mode == "all":
        n_new = h_new.get()
        print(f"  New sources: {n_new} pairs")

    if mode in ("merge", "all"):
        print("--- MERGING + PUSH TO HF ---")
        n_total = merge_and_push.remote()
        print(f"  Total merged: {n_total} pairs")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done (mode={mode})")
