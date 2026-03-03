"""
SCAR Data Pipeline — Modal Labs
==========================================
Processes 9 data sources into 3 HuggingFace datasets:
  1. scar-corpus        (~250k Solidity contracts)
  2. scar-pairs  (~33k query-code pairs)
  3. scar-eval               (~500-700 held-out findings)

Usage:
  modal run scripts/pipeline.py
  modal run scripts/pipeline.py --wave 1   # only Wave 1
  modal run scripts/pipeline.py --wave 2   # only Wave 2 (needs Wave 1 done)
  modal run scripts/pipeline.py --wave 3   # only Wave 3 (needs Wave 2 done)
"""

import modal
import os

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("scar-pipeline")

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
        "pyyaml>=6.0",
    )
)

INTERMEDIATE_DIR = "/intermediates"
HF_SECRET = modal.Secret.from_name("huggingface-token")

# Change this to your HuggingFace username
HF_USERNAME = "Farseen0"


# ---------------------------------------------------------------------------
# Shared utility functions (available to all Modal functions)
# ---------------------------------------------------------------------------
import re
import hashlib


def normalize_solidity(source: str) -> str:
    """Strip comments and collapse whitespace for dedup hashing."""
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    source = re.sub(r'//.*$', '', source, flags=re.MULTILINE)
    source = re.sub(r'SPDX-License-Identifier:.*$', '', source, flags=re.MULTILINE)
    source = re.sub(r'pragma\s+solidity.*?;', '', source)
    lines = [line.strip() for line in source.splitlines()]
    lines = [line for line in lines if line]
    return '\n'.join(lines)


def hash_contract(source: str) -> str:
    """SHA256 hash of normalized source for dedup."""
    normalized = normalize_solidity(source)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def count_lines(source: str) -> int:
    """Count non-empty lines in raw source."""
    return len([line for line in source.splitlines() if line.strip()])


def is_import_only(source: str, threshold: float = 0.8) -> bool:
    """Check if >threshold of non-empty lines are import statements."""
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if not lines:
        return True
    import_lines = [l for l in lines if l.startswith('import ') or l.startswith('import"')]
    return len(import_lines) / len(lines) > threshold


def passes_quality_filter(source: str, min_lines: int = 50) -> bool:
    """Returns True if contract should be KEPT."""
    if count_lines(source) < min_lines:
        return False
    if is_import_only(source):
        return False
    return True


def log(msg: str):
    """Timestamped logging."""
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ═══════════════════════════════════════════════════════════════════════════
# WAVE 1: Source Processing (9 functions, run in parallel)
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 01 — DISL
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=14400)
def process_disl():
    """Load DISL decomposed dataset, quality filter, hash dedup, sample 120k."""
    import pandas as pd
    import random
    from tqdm import tqdm
    from datasets import load_dataset

    output_path = f"{INTERMEDIATE_DIR}/disl_processed.parquet"
    if os.path.exists(output_path):
        log("DISL: already exists, skipping")
        return

    log("DISL: loading dataset...")
    try:
        ds = load_dataset('ASSERT-KTH/DISL', 'decomposed', split='train')
        use_streaming = False
    except Exception:
        ds = load_dataset('ASSERT-KTH/DISL', 'decomposed', split='train', streaming=True)
        use_streaming = True

    random.seed(42)
    seen_hashes = set()
    records = []
    stats = {'total': 0, 'too_short': 0, 'import_only': 0, 'duplicate': 0, 'no_source': 0, 'kept': 0}

    for row in tqdm(ds, desc='Processing DISL'):
        stats['total'] += 1
        source_code = row.get('source_code', '')

        if not source_code or not source_code.strip():
            stats['no_source'] += 1
            continue

        num_lines = count_lines(source_code)
        if num_lines < 50:
            stats['too_short'] += 1
            continue

        if is_import_only(source_code):
            stats['import_only'] += 1
            continue

        contract_hash = hash_contract(source_code)
        if contract_hash in seen_hashes:
            stats['duplicate'] += 1
            continue
        seen_hashes.add(contract_hash)

        records.append({
            'source': 'DISL',
            'contract_code': source_code,
            'num_lines': num_lines,
            'has_vuln_labels': False,
            'vuln_labels': '',
            'contract_hash': contract_hash,
        })
        stats['kept'] += 1

        if stats['kept'] % 50000 == 0:
            log(f"  DISL: kept {stats['kept']} so far (processed {stats['total']})")

    df = pd.DataFrame(records)

    TARGET_SAMPLE = 120_000
    if len(df) > TARGET_SAMPLE:
        df = df.sample(n=TARGET_SAMPLE, random_state=42)
        log(f"DISL: sampled down to {TARGET_SAMPLE}")

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    df.to_parquet(output_path, index=False)
    vol.commit()
    log(f"DISL: saved {len(df)} contracts. Stats: {stats}")


# ---------------------------------------------------------------------------
# 02 — slither-audited
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=14400)
def process_slither():
    """Load slither-audited big-multilabel, quality filter, extract labels."""
    import pandas as pd
    from tqdm import tqdm
    from datasets import load_dataset, concatenate_datasets

    output_path = f"{INTERMEDIATE_DIR}/slither_audited_processed.parquet"
    if os.path.exists(output_path):
        log("Slither: already exists, skipping")
        return

    log("Slither: loading dataset (big-multilabel)...")
    ds = load_dataset('mwritescode/slither-audited-smart-contracts', 'big-multilabel', trust_remote_code=True)
    all_splits = [ds[split] for split in ds.keys()]
    merged = concatenate_datasets(all_splits)
    log(f"Slither: loaded {len(merged)} rows from {len(ds)} splits")

    # big-multilabel config encodes 38 slither detectors into 6 multi-label classes.
    # PRD says "38 → 9" but the dataset's actual encoding uses these 6 categories.
    LABEL_MAP = {
        0: 'access-control', 1: 'arithmetic', 2: 'other',
        3: 'reentrancy', 4: 'safe', 5: 'unchecked-calls',
    }

    records = []
    stats = {'total': len(merged), 'too_short': 0, 'import_only': 0, 'no_source': 0, 'kept': 0}

    for i in tqdm(range(len(merged)), desc='Processing slither-audited'):
        row = merged[i]
        source_code = row.get('source_code', '')

        if not source_code or not source_code.strip():
            stats['no_source'] += 1
            continue

        num_lines = count_lines(source_code)
        if num_lines < 50:
            stats['too_short'] += 1
            continue

        if is_import_only(source_code):
            stats['import_only'] += 1
            continue

        slither_labels = row.get('slither', [])
        if isinstance(slither_labels, (list, tuple)):
            vuln_labels = [LABEL_MAP.get(l, f'unknown-{l}') for l in slither_labels if l != 4]
        else:
            vuln_labels = []

        records.append({
            'source': 'slither-audited',
            'contract_code': source_code,
            'num_lines': num_lines,
            'has_vuln_labels': len(vuln_labels) > 0,
            'vuln_labels': '|'.join(vuln_labels) if vuln_labels else '',
            'contract_hash': hash_contract(source_code),
        })
        stats['kept'] += 1

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset='contract_hash', keep='first')

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    df.to_parquet(output_path, index=False)
    vol.commit()
    log(f"Slither: saved {len(df)} contracts. Stats: {stats}")


# ---------------------------------------------------------------------------
# 03 — DeFiVulnLabs
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=7200)
def process_defivulnlabs():
    """Clone DeFiVulnLabs, extract .sol files, tag vuln types from filename.

    Note: No 50-line filter applied here. DeFiVulnLabs files are intentionally
    short (~20-40 lines) vulnerability pattern labs. Exempted at merge step too.
    """
    import subprocess
    import glob
    import pandas as pd

    output_path = f"{INTERMEDIATE_DIR}/defivulnlabs_processed.parquet"
    if os.path.exists(output_path):
        log("DeFiVulnLabs: already exists, skipping")
        return

    log("DeFiVulnLabs: cloning repo...")
    clone_dir = "/tmp/DeFiVulnLabs"
    subprocess.run(["git", "clone", "--depth", "1",
                     "https://github.com/SunWeb3Sec/DeFiVulnLabs.git", clone_dir],
                    check=True, capture_output=True)

    sol_files = glob.glob(f'{clone_dir}/src/**/*.sol', recursive=True)
    sol_files += glob.glob(f'{clone_dir}/test/**/*.sol', recursive=True)
    sol_files = list(set(sol_files))
    log(f"DeFiVulnLabs: found {len(sol_files)} .sol files")

    records = []
    for fpath in sorted(sol_files):
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                source_code = f.read()
        except Exception:
            continue

        if not source_code.strip():
            continue

        filename = os.path.basename(fpath).replace('.sol', '')
        vuln_type = filename.replace('_', ' ').replace('-', ' ')
        num_lines = count_lines(source_code)

        records.append({
            'source': 'DeFiVulnLabs',
            'contract_code': source_code,
            'num_lines': num_lines,
            'has_vuln_labels': True,
            'vuln_labels': vuln_type,
            'contract_hash': hash_contract(source_code),
        })

    df = pd.DataFrame(records)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    df.to_parquet(output_path, index=False)
    vol.commit()
    log(f"DeFiVulnLabs: saved {len(df)} contracts")


# ---------------------------------------------------------------------------
# 04 — FORGE-Curated (ALL 3 STREAMS)
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=14400)
def process_forge():
    """Clone FORGE-Curated, temporal split, output Stream 1/2/3 + metadata."""
    import subprocess
    import glob
    import json
    import random
    import pandas as pd
    from collections import defaultdict
    from tqdm import tqdm

    # Check if ALL outputs exist
    outputs = [
        f"{INTERMEDIATE_DIR}/forge_stream1_contracts.parquet",
        f"{INTERMEDIATE_DIR}/forge_stream2_pairs.parquet",
        f"{INTERMEDIATE_DIR}/forge_stream3_eval.parquet",
        f"{INTERMEDIATE_DIR}/forge_split_metadata.json",
    ]
    if all(os.path.exists(p) for p in outputs):
        log("FORGE: all outputs exist, skipping")
        return

    log("FORGE: cloning repo...")
    clone_dir = "/tmp/FORGE-Curated"
    subprocess.run(["git", "clone", "--depth", "1",
                     "https://github.com/shenyimings/FORGE-Curated.git", clone_dir],
                    check=True, capture_output=True)

    # ===== PHASE 1: Load all findings and extract audit_date =====
    findings_dir = f'{clone_dir}/dataset-curated/findings'
    findings_json_files = glob.glob(f'{findings_dir}/**/*.json', recursive=True)
    log(f"FORGE: found {len(findings_json_files)} finding JSON files")

    reports = []
    for jpath in findings_json_files:
        try:
            with open(jpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        project_info = data.get('project_info', {})
        audit_date = project_info.get('audit_date', 'n/a')
        report_path = data.get('path', '')

        firm = ''
        if report_path:
            parts = report_path.replace('\\', '/').split('/')
            for i, part in enumerate(parts):
                if part == 'reports' and i + 1 < len(parts):
                    firm = parts[i + 1]
                    break

        report_name = os.path.basename(jpath).replace('.json', '')
        findings_list = data.get('findings', [])

        reports.append({
            'report_name': report_name,
            'report_path': report_path,
            'json_path': jpath,
            'audit_date': audit_date,
            'firm': firm,
            'num_findings': len(findings_list),
            'findings': findings_list,
            'project_info': project_info,
        })

    log(f"FORGE: loaded {len(reports)} reports")

    # ===== PHASE 2: Temporal split by audit_date =====
    reports_with_dates = [r for r in reports if r['audit_date'] != 'n/a' and r['audit_date']]
    reports_no_dates = [r for r in reports if r['audit_date'] == 'n/a' or not r['audit_date']]

    reports_with_dates.sort(key=lambda r: r['audit_date'])

    n = len(reports_with_dates)
    split_60 = int(n * 0.60)
    split_80 = int(n * 0.80)

    stream1_reports = reports_no_dates + reports_with_dates[:split_60]
    stream2_reports = reports_with_dates[split_60:split_80]
    stream3_reports = reports_with_dates[split_80:]

    # Verify no overlap
    s1_names = {r['report_name'] for r in stream1_reports}
    s2_names = {r['report_name'] for r in stream2_reports}
    s3_names = {r['report_name'] for r in stream3_reports}
    assert s1_names.isdisjoint(s2_names), 'LEAKAGE: Stream 1 and 2 share reports!'
    assert s1_names.isdisjoint(s3_names), 'LEAKAGE: Stream 1 and 3 share reports!'
    assert s2_names.isdisjoint(s3_names), 'LEAKAGE: Stream 2 and 3 share reports!'
    log(f"FORGE split: S1={len(stream1_reports)} reports, S2={len(stream2_reports)}, S3={len(stream3_reports)}")

    # ===== PHASE 3: Build Stream 1 — SAE Pretraining Corpus =====
    # Collects ALL .sol files from contracts dir (not split by report date).
    # SAE pretraining is unsupervised — no vulnerability labels leak.
    # Any hash overlap with Stream 3 is removed later in merge_stream1() safety net.
    contracts_dir = f'{clone_dir}/dataset-curated/contracts'
    all_sol_files = glob.glob(f'{contracts_dir}/**/*.sol', recursive=True)

    seen_hashes = set()
    stream1_records = []
    stats = {'total': 0, 'too_short': 0, 'import_only': 0, 'duplicate': 0, 'kept': 0}

    for fpath in tqdm(all_sol_files, desc='FORGE Stream 1 contracts'):
        stats['total'] += 1
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                source_code = f.read()
        except Exception:
            continue

        if not source_code.strip():
            continue

        num_lines = count_lines(source_code)
        if num_lines < 50:
            stats['too_short'] += 1
            continue

        if is_import_only(source_code):
            stats['import_only'] += 1
            continue

        contract_hash = hash_contract(source_code)
        if contract_hash in seen_hashes:
            stats['duplicate'] += 1
            continue
        seen_hashes.add(contract_hash)

        stream1_records.append({
            'source': 'FORGE-Curated',
            'contract_code': source_code,
            'num_lines': num_lines,
            'has_vuln_labels': False,
            'vuln_labels': '',
            'contract_hash': contract_hash,
        })
        stats['kept'] += 1

    df_stream1 = pd.DataFrame(stream1_records)
    log(f"FORGE Stream 1: {len(df_stream1)} contracts. Stats: {stats}")

    # ===== PHASE 4: Load VFP files for Stream 2 and Stream 3 =====
    vfp_dir = f'{clone_dir}/flatten/vfp-vuln'
    vfp_files = glob.glob(f'{vfp_dir}/*.json')
    if not vfp_files:
        vfp_dir = f'{clone_dir}/flatten/vfp'
        vfp_files = glob.glob(f'{vfp_dir}/*.json')

    vfps = []
    for vfp_path in tqdm(vfp_files, desc='Loading VFPs'):
        try:
            with open(vfp_path, 'r', encoding='utf-8') as f:
                vfp = json.load(f)
            vfps.append(vfp)
        except Exception:
            continue

    log(f"FORGE: loaded {len(vfps)} VFP files")

    # ===== PHASE 5: Assign VFPs to streams =====
    report_to_stream = {}
    for r in stream1_reports:
        report_to_stream[r['report_name']] = 1
    for r in stream2_reports:
        report_to_stream[r['report_name']] = 2
    for r in stream3_reports:
        report_to_stream[r['report_name']] = 3

    stream2_vfps = []
    stream3_vfps = []

    for vfp in vfps:
        proj_name = vfp.get('project_name', '')
        stream = report_to_stream.get(proj_name.replace('.pdf', ''), None)

        if stream is None:
            proj_clean = proj_name.lower().replace(' ', '').replace('-', '').replace('_', '').replace('.pdf', '')
            for rname, s in report_to_stream.items():
                rname_clean = rname.lower().replace(' ', '').replace('-', '').replace('_', '')
                if proj_clean in rname_clean or rname_clean in proj_clean:
                    stream = s
                    break

        if stream == 2:
            stream2_vfps.append(vfp)
        elif stream == 3:
            stream3_vfps.append(vfp)
        # Stream 1 reports contribute contracts only, NOT pairs (PRD: oldest 60% → Stream 1)

    # ===== PHASE 6: Build Stream 2 contrastive pairs =====
    # quality_tier: 1=gold (SmartBugs Curated), 2=high (Solodit/FORGE/DeFiHackLabs), 3=good (msc)
    random.seed(42)
    pairs = []
    all_findings_with_code = []

    for vfp in stream2_vfps:
        affected_files = vfp.get('affected_files', {})
        all_code = '\n'.join(affected_files.values())

        for finding in vfp.get('findings', []):
            severity = finding.get('severity', '')
            if isinstance(severity, list):
                severity = severity[0] if severity else ''
            severity = str(severity).strip()

            if severity.lower() not in ('medium', 'high', 'critical'):
                continue

            title = finding.get('title', '')
            description = finding.get('description', '')
            vuln_type = ''
            category = finding.get('category', {})
            if category and isinstance(category, dict):
                cwes = []
                for level_cwes in category.values():
                    if isinstance(level_cwes, list):
                        cwes.extend(level_cwes)
                vuln_type = '|'.join(cwes[:3])

            finding_files = finding.get('files', [])
            if finding_files and isinstance(finding_files, list):
                matched_code = []
                for ff in finding_files:
                    fname = os.path.basename(ff)
                    if fname in affected_files:
                        matched_code.append(affected_files[fname])
                positive_code = '\n'.join(matched_code) if matched_code else all_code
            else:
                positive_code = all_code

            if not positive_code.strip():
                continue

            sev_upper = severity.upper()
            if sev_upper == 'CRITICAL':
                sev_upper = 'HIGH'

            desc_truncated = description[:300].strip()
            if len(description) > 300:
                desc_truncated += '...'
            query = f'{sev_upper} severity: {title}. {desc_truncated}'

            entry = {
                'query': query,
                'positive': positive_code,
                'severity': sev_upper,
                'vuln_type': vuln_type,
                'title': title,
                'report_name': vfp.get('project_name', ''),
            }
            all_findings_with_code.append(entry)

    by_report = defaultdict(list)
    for entry in all_findings_with_code:
        by_report[entry['report_name']].append(entry)

    all_positives = [e['positive'] for e in all_findings_with_code]

    for entry in tqdm(all_findings_with_code, desc='Building FORGE pairs'):
        same_report = by_report[entry['report_name']]
        candidates = [e for e in same_report if e['title'] != entry['title'] and e['positive'].strip()]

        if candidates:
            hard_neg = random.choice(candidates)['positive']
            neg_type = 'same_report_diff_finding'
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
            'source': 'FORGE-Curated',
            'severity': entry['severity'],
            'vuln_type': entry['vuln_type'],
            'quality_tier': 2,
        })

    df_stream2 = pd.DataFrame(pairs)
    log(f"FORGE Stream 2: {len(df_stream2)} pairs")

    # ===== PHASE 7: Build Stream 3 eval set =====
    # Build report→metadata lookup for audit_firm and report_date
    report_metadata = {}
    for r in reports:
        report_metadata[r['report_name']] = {
            'firm': r.get('firm', ''),
            'audit_date': r.get('audit_date', ''),
        }

    eval_records = []

    for vfp in stream3_vfps:
        affected_files = vfp.get('affected_files', {})
        all_code = '\n'.join(affected_files.values())

        for finding in vfp.get('findings', []):
            severity = finding.get('severity', '')
            if isinstance(severity, list):
                severity = severity[0] if severity else ''
            severity = str(severity).strip()

            if severity.lower() not in ('medium', 'high', 'critical'):
                continue

            title = finding.get('title', '')
            description = finding.get('description', '')
            vuln_type = ''
            category = finding.get('category', {})
            if category and isinstance(category, dict):
                cwes = []
                for level_cwes in category.values():
                    if isinstance(level_cwes, list):
                        cwes.extend(level_cwes)
                vuln_type = '|'.join(cwes[:3])

            finding_files = finding.get('files', [])
            if finding_files and isinstance(finding_files, list):
                matched_code = []
                for ff in finding_files:
                    fname = os.path.basename(ff)
                    if fname in affected_files:
                        matched_code.append(affected_files[fname])
                ground_truth_code = '\n'.join(matched_code) if matched_code else all_code
            else:
                ground_truth_code = all_code

            if not ground_truth_code.strip():
                continue

            sev_upper = severity.upper()
            if sev_upper == 'CRITICAL':
                sev_upper = 'HIGH'

            desc_truncated = description[:300].strip()
            if len(description) > 300:
                desc_truncated += '...'

            # Look up audit firm and date from report metadata
            proj_name = vfp.get('project_name', '')
            rpt_meta = report_metadata.get(
                proj_name.replace('.pdf', ''),
                {'firm': '', 'audit_date': ''}
            )

            eval_records.append({
                'query': f'{sev_upper} severity: {title}. {desc_truncated}',
                'ground_truth_code': ground_truth_code,
                'severity': sev_upper,
                'vuln_type': vuln_type,
                'report_name': proj_name,
                'audit_firm': rpt_meta['firm'],
                'report_date': rpt_meta['audit_date'],
            })

    df_stream3 = pd.DataFrame(eval_records)
    log(f"FORGE Stream 3: {len(df_stream3)} eval findings")

    # ===== Save all outputs =====
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    df_stream1.to_parquet(f"{INTERMEDIATE_DIR}/forge_stream1_contracts.parquet", index=False)
    df_stream2.to_parquet(f"{INTERMEDIATE_DIR}/forge_stream2_pairs.parquet", index=False)
    df_stream3.to_parquet(f"{INTERMEDIATE_DIR}/forge_stream3_eval.parquet", index=False)

    # Save split metadata
    split_meta = {
        'stream1_reports': sorted(list(s1_names)),
        'stream2_reports': sorted(list(s2_names)),
        'stream3_reports': sorted(list(s3_names)),
        'split_method': 'temporal_by_audit_date',
        'split_ratios': '60/20/20',
        'total_reports': len(reports),
        'reports_with_dates': len(reports_with_dates),
        'reports_no_dates': len(reports_no_dates),
    }
    with open(f"{INTERMEDIATE_DIR}/forge_split_metadata.json", 'w') as f:
        json.dump(split_meta, f, indent=2)

    vol.commit()
    log("FORGE: all 4 outputs saved")


# ---------------------------------------------------------------------------
# 05 — SmartBugs Wild
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=14400)
def process_smartbugs_wild():
    """Clone smartbugs-wild, extract .sol files, quality filter, dedup."""
    import subprocess
    import glob
    import pandas as pd
    from tqdm import tqdm

    output_path = f"{INTERMEDIATE_DIR}/smartbugs_wild_processed.parquet"
    if os.path.exists(output_path):
        log("SmartBugs-Wild: already exists, skipping")
        return

    log("SmartBugs-Wild: cloning repo...")
    clone_dir = "/tmp/smartbugs-wild"
    subprocess.run(["git", "clone", "--depth", "1",
                     "https://github.com/smartbugs/smartbugs-wild.git", clone_dir],
                    check=True, capture_output=True)

    sol_files = glob.glob(f'{clone_dir}/**/*.sol', recursive=True)
    log(f"SmartBugs-Wild: found {len(sol_files)} .sol files")

    seen_hashes = set()
    records = []
    stats = {'total': 0, 'too_short': 0, 'import_only': 0, 'duplicate': 0, 'read_error': 0, 'kept': 0}

    for fpath in tqdm(sol_files, desc='Processing SmartBugs Wild'):
        stats['total'] += 1
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                source_code = f.read()
        except Exception:
            stats['read_error'] += 1
            continue

        if not source_code.strip():
            stats['read_error'] += 1
            continue

        num_lines = count_lines(source_code)
        if num_lines < 50:
            stats['too_short'] += 1
            continue

        if is_import_only(source_code):
            stats['import_only'] += 1
            continue

        contract_hash = hash_contract(source_code)
        if contract_hash in seen_hashes:
            stats['duplicate'] += 1
            continue
        seen_hashes.add(contract_hash)

        records.append({
            'source': 'SmartBugs-Wild',
            'contract_code': source_code,
            'num_lines': num_lines,
            'has_vuln_labels': False,
            'vuln_labels': '',
            'contract_hash': contract_hash,
        })
        stats['kept'] += 1

    df = pd.DataFrame(records)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    df.to_parquet(output_path, index=False)
    vol.commit()
    log(f"SmartBugs-Wild: saved {len(df)} contracts. Stats: {stats}")


# ---------------------------------------------------------------------------
# 06 — Solodit (HARDEST — multi-firm markdown parser)
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=21600, cpu=4.0)
def process_solodit():
    """Clone solodit_content, parse 16+ audit firm markdown formats, build pairs."""
    import subprocess
    import glob
    import yaml
    import random
    import pandas as pd
    from typing import Optional
    from tqdm import tqdm
    from collections import defaultdict

    output_path = f"{INTERMEDIATE_DIR}/solodit_pairs.parquet"
    if os.path.exists(output_path):
        log("Solodit: already exists, skipping")
        return

    log("Solodit: cloning repo...")
    clone_dir = "/tmp/solodit_content"
    subprocess.run(["git", "clone", "--depth", "1",
                     "https://github.com/solodit/solodit_content.git", clone_dir],
                    check=True, capture_output=True)

    reports_dir = f'{clone_dir}/reports'
    firms = sorted([d for d in os.listdir(reports_dir)
                     if os.path.isdir(os.path.join(reports_dir, d))])

    firm_files = {}
    total_md = 0
    for firm in firms:
        md_files = glob.glob(os.path.join(reports_dir, firm, '**/*.md'), recursive=True)
        firm_files[firm] = md_files
        total_md += len(md_files)
    log(f"Solodit: found {len(firms)} firms, {total_md} markdown files")

    # --- Severity detection ---
    SEVERITY_PATTERNS = [
        (r'(?i)\b(?:severity|risk)\s*[:\-]?\s*(critical|high|medium|low|informational|info)', None),
        (r'(?i)\[?(H|HIGH|CRITICAL|C)[-_]?\d+\]?', 'HIGH'),
        (r'(?i)\[?(M|MEDIUM|MED)[-_]?\d+\]?', 'MEDIUM'),
        (r'(?i)\[?(L|LOW)[-_]?\d+\]?', 'LOW'),
        (r'(?i)\[?(I|INFO|INFORMATIONAL)[-_]?\d+\]?', 'INFORMATIONAL'),
        (r'\U0001F534|\u274C', 'HIGH'),
        (r'\U0001F7E0|\u26A0', 'MEDIUM'),
        (r'\U0001F7E1', 'LOW'),
        (r'\U0001F535', 'INFORMATIONAL'),
    ]

    def detect_severity(text: str) -> Optional[str]:
        text_clean = text.strip()
        for pattern, fixed_sev in SEVERITY_PATTERNS:
            match = re.search(pattern, text_clean)
            if match:
                if fixed_sev:
                    return fixed_sev
                sev = match.group(1).strip().upper()
                sev_map = {
                    'CRITICAL': 'HIGH', 'HIGH': 'HIGH', 'H': 'HIGH',
                    'MEDIUM': 'MEDIUM', 'MED': 'MEDIUM', 'M': 'MEDIUM',
                    'LOW': 'LOW', 'L': 'LOW',
                    'INFORMATIONAL': 'INFORMATIONAL', 'INFO': 'INFORMATIONAL', 'I': 'INFORMATIONAL',
                }
                return sev_map.get(sev, None)
        return None

    def extract_code_blocks(text: str) -> list:
        pattern = r'```(?:solidity|sol|js|javascript)?\s*\n(.*?)```'
        blocks = re.findall(pattern, text, re.DOTALL)
        sol_keywords = ['function ', 'contract ', 'pragma ', 'mapping(', 'address ',
                        'uint', 'require(', 'emit ', 'modifier ']
        sol_blocks = []
        for block in blocks:
            if any(kw in block for kw in sol_keywords):
                sol_blocks.append(block.strip())
            elif len(block.strip()) > 50:
                sol_blocks.append(block.strip())
        return sol_blocks if sol_blocks else blocks

    def extract_frontmatter(content: str) -> dict:
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    return yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    return {}
        return {}

    # Severity category patterns for ## headings (e.g., "## High Risk", "## Critical Risk")
    SEVERITY_CATEGORY_MAP = {
        r'(?i)^\s*(?:critical|crit)\s*(?:risk|severity)?\s*$': 'HIGH',
        r'(?i)^\s*high\s*(?:risk|severity)?\s*$': 'HIGH',
        r'(?i)^\s*medium\s*(?:risk|severity)?\s*$': 'MEDIUM',
        r'(?i)^\s*low\s*(?:risk|severity)?\s*$': 'LOW',
        r'(?i)^\s*informational?\s*(?:risk|severity)?\s*$': 'INFORMATIONAL',
        r'(?i)^\s*info\s*(?:risk|severity)?\s*$': 'INFORMATIONAL',
        r'(?i)^\s*gas\s*(?:optimization|improvement)s?\s*$': None,  # skip gas
    }

    def heading_is_severity_category(heading_text: str):
        """Check if a ## heading is a severity category like 'High Risk'.
        Returns severity string or None."""
        for pattern, sev in SEVERITY_CATEGORY_MAP.items():
            if re.match(pattern, heading_text.strip()):
                return sev
        return None

    def extract_code_blocks_v2(text: str) -> list:
        """Extract code blocks — fenced, blockquoted, and indented."""
        sol_keywords = ['function ', 'contract ', 'pragma ', 'mapping(', 'address ',
                        'uint', 'require(', 'emit ', 'modifier ', 'interface ',
                        'library ', 'event ', 'struct ', 'enum ', 'constructor(']
        blocks = []

        # 1. Fenced code blocks (```...```) — any language tag
        fenced = re.findall(r'```[a-z]*\s*\n(.*?)```', text, re.DOTALL)
        for block in fenced:
            block = block.strip()
            if any(kw in block for kw in sol_keywords) or len(block) > 50:
                blocks.append(block)

        # 2. Indented code blocks (4+ spaces or tab, consecutive lines)
        if not blocks:
            indented_lines = []
            for line in text.splitlines():
                if re.match(r'^(?:    |\t)', line):
                    indented_lines.append(line.strip())
                else:
                    if indented_lines and len(indented_lines) >= 3:
                        candidate = '\n'.join(indented_lines)
                        if any(kw in candidate for kw in sol_keywords):
                            blocks.append(candidate)
                    indented_lines = []
            if indented_lines and len(indented_lines) >= 3:
                candidate = '\n'.join(indented_lines)
                if any(kw in candidate for kw in sol_keywords):
                    blocks.append(candidate)

        return blocks

    def parse_markdown_findings(content: str, firm: str, filename: str) -> list:
        """Parse audit report markdown into individual findings.

        Uses hierarchical heading parsing:
        - ## headings set severity context (e.g., "## High Risk")
        - ### headings are individual findings that inherit parent severity
        - Also detects severity from heading text or body **Severity** field
        """
        frontmatter = extract_frontmatter(content)
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                content = parts[2]

        # Parse into hierarchical structure: ## categories containing ### findings
        current_h2_severity = None  # severity from parent ## category heading
        current_h3_heading = None
        current_h3_body = []
        findings_raw = []  # list of (heading, body, inherited_severity)

        for line in content.splitlines():
            # Check for ## heading (severity category or finding)
            h2_match = re.match(r'^##\s+(.+)$', line)
            if h2_match:
                # Save previous ### finding if any
                if current_h3_heading is not None:
                    findings_raw.append((current_h3_heading, '\n'.join(current_h3_body), current_h2_severity))
                    current_h3_heading = None
                    current_h3_body = []

                h2_text = h2_match.group(1).strip()
                cat_sev = heading_is_severity_category(h2_text)
                if cat_sev is not None or re.match(r'(?i)^\s*gas', h2_text):
                    # This is a category heading — set context
                    current_h2_severity = cat_sev
                else:
                    # This ## is itself a finding (some firms use ## for findings)
                    current_h3_heading = h2_text
                    current_h3_body = []
                continue

            # Check for ### or #### heading (individual finding)
            h34_match = re.match(r'^#{3,4}\s+(.+)$', line)
            if h34_match:
                # Save previous finding
                if current_h3_heading is not None:
                    findings_raw.append((current_h3_heading, '\n'.join(current_h3_body), current_h2_severity))
                current_h3_heading = h34_match.group(1).strip()
                current_h3_body = []
                continue

            # Body line
            current_h3_body.append(line)

        # Don't forget last section
        if current_h3_heading is not None:
            findings_raw.append((current_h3_heading, '\n'.join(current_h3_body), current_h2_severity))

        # Process each finding
        findings = []
        for heading, body, inherited_severity in findings_raw:
            # Determine severity (priority: heading > body **Severity** field > inherited from ##)
            severity = detect_severity(heading)
            if severity is None:
                # Check for **Severity**: Critical/High/Medium/Low in body
                sev_field = re.search(r'\*\*Severity\*\*\s*[:\-]?\s*(Critical|High|Medium|Low|Informational)',
                                      body[:500], re.IGNORECASE)
                if sev_field:
                    sev_val = sev_field.group(1).upper()
                    severity = {'CRITICAL': 'HIGH', 'HIGH': 'HIGH', 'MEDIUM': 'MEDIUM',
                                'LOW': 'LOW', 'INFORMATIONAL': 'INFORMATIONAL'}.get(sev_val)
            if severity is None:
                severity = detect_severity(body[:500])
            if severity is None:
                severity = inherited_severity

            # Include HIGH, MEDIUM, and LOW (LOW adds volume for contrastive training)
            if severity not in ('HIGH', 'MEDIUM', 'LOW'):
                continue

            title = re.sub(r'(?i)\[?[HMCLhmlc][-_]?\d+\]?\s*[-:.]?\s*', '', heading).strip()
            title = re.sub(r'^\d+\.\s*', '', title).strip()  # strip numbered prefixes
            if not title:
                title = heading

            code_blocks = extract_code_blocks_v2(body)
            code_snippet = '\n\n'.join(code_blocks[:3])

            desc_text = re.sub(r'```.*?```', '', body, flags=re.DOTALL)
            desc_text = re.sub(r'\n{3,}', '\n\n', desc_text).strip()
            description = desc_text[:300].strip()
            if len(desc_text) > 300:
                last_space = description.rfind(' ')
                if last_space > 200:
                    description = description[:last_space]
                description += '...'

            date_match = re.match(r'(\d{4}-\d{2}-\d{2})', filename)
            date = date_match.group(1) if date_match else ''

            protocol = filename
            if date:
                protocol = filename[len(date):].strip('-_ ')
            protocol = protocol.replace('.md', '').replace('-', ' ').replace('_', ' ').strip()

            findings.append({
                'title': title,
                'severity': severity,
                'description': description,
                'code_snippet': code_snippet,
                'firm': firm,
                'date': date,
                'protocol': protocol,
                'filename': filename,
                'has_code': len(code_snippet) > 0,
            })

        return findings

    # --- Parse all firms ---
    all_findings = []
    firm_stats = defaultdict(lambda: {'files': 0, 'parsed': 0, 'findings': 0, 'with_code': 0, 'errors': 0})

    for firm in firms:
        files = firm_files[firm]
        for fpath in tqdm(files, desc=f'Parsing {firm}', leave=False):
            firm_stats[firm]['files'] += 1
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except Exception:
                firm_stats[firm]['errors'] += 1
                continue

            if not content.strip() or len(content) < 100:
                continue

            filename = os.path.basename(fpath)
            findings = parse_markdown_findings(content, firm, filename)

            if findings:
                firm_stats[firm]['parsed'] += 1
                firm_stats[firm]['findings'] += len(findings)
                firm_stats[firm]['with_code'] += sum(1 for f in findings if f['has_code'])
                all_findings.extend(findings)

    log(f"Solodit: extracted {len(all_findings)} findings from {len(firms)} firms")

    # --- Build contrastive pairs ---
    random.seed(42)

    by_firm = defaultdict(list)
    for f in all_findings:
        by_firm[f['firm']].append(f)

    all_codes = [f['code_snippet'] for f in all_findings if f['code_snippet']]

    pairs = []
    skipped_no_code = 0

    for finding in tqdm(all_findings, desc='Building Solodit pairs'):
        # ONLY use code as positive — never description (causes query-positive leakage)
        if not finding['code_snippet'] or not finding['code_snippet'].strip():
            skipped_no_code += 1
            continue
        positive = finding['code_snippet']

        desc_truncated = finding['description'][:300]
        if len(finding['description']) > 300:
            desc_truncated += '...'
        query = f'{finding["severity"]} severity: {finding["title"]}. {desc_truncated}'

        same_firm = by_firm[finding['firm']]
        # Hard negatives: prefer code-only too (avoid type mismatch shortcut)
        candidates = [
            f for f in same_firm
            if f['protocol'] != finding['protocol']
            and f['title'] != finding['title']
            and f['code_snippet']  # require code for hard negatives too
        ]

        if candidates:
            neg_finding = random.choice(candidates)
            hard_neg = neg_finding['code_snippet']
            neg_type = 'same_firm_diff_protocol'
        elif all_codes:
            hard_neg = random.choice(all_codes)
            neg_type = 'random'
        else:
            hard_neg = ''
            neg_type = 'none'

        pairs.append({
            'query': query,
            'positive': positive,
            'hard_negative': hard_neg,
            'negative_type': neg_type,
            'source': 'Solodit',
            'severity': finding['severity'],
            'vuln_type': '',
            'quality_tier': 2,
        })

    pairs_df = pd.DataFrame(pairs)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    pairs_df.to_parquet(output_path, index=False)
    vol.commit()
    log(f"Solodit: saved {len(pairs_df)} pairs (skipped {skipped_no_code} with no code)")

    # Log firm stats
    for firm, st in sorted(firm_stats.items()):
        log(f"  {firm}: {st['files']} files, {st['parsed']} parsed, "
            f"{st['findings']} findings, {st['with_code']} with code")


# ---------------------------------------------------------------------------
# 07 — DeFiHackLabs
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=7200)
def process_defihacklabs():
    """Clone DeFiHackLabs, parse *_exp.sol exploit POC files, build pairs."""
    import subprocess
    import glob
    import random
    import pandas as pd
    from tqdm import tqdm
    from collections import defaultdict

    output_path = f"{INTERMEDIATE_DIR}/defihacklabs_pairs.parquet"
    if os.path.exists(output_path):
        log("DeFiHackLabs: already exists, skipping")
        return

    log("DeFiHackLabs: cloning repo...")
    clone_dir = "/tmp/DeFiHackLabs"
    subprocess.run(["git", "clone", "--depth", "1",
                     "https://github.com/SunWeb3Sec/DeFiHackLabs.git", clone_dir],
                    check=True, capture_output=True)

    exp_files = glob.glob(f'{clone_dir}/src/test/**/*_exp.sol', recursive=True)
    log(f"DeFiHackLabs: found {len(exp_files)} exploit files")

    vuln_keywords = {
        'reentrancy': 'reentrancy', 'reentrant': 'reentrancy', 're-entrancy': 'reentrancy',
        'flash loan': 'flash-loan', 'flashloan': 'flash-loan',
        'price manipulation': 'price-manipulation', 'oracle manipulation': 'oracle-manipulation',
        'access control': 'access-control', 'unauthorized': 'access-control',
        'overflow': 'arithmetic', 'underflow': 'arithmetic',
        'delegatecall': 'delegatecall', 'storage collision': 'storage-collision',
        'front.?run': 'front-running', 'sandwich': 'front-running',
        'rug.?pull': 'rug-pull',
    }

    def parse_exploit_file(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception:
            return None

        if not content.strip():
            return None

        filename = os.path.basename(filepath)
        protocol = filename.replace('_exp.sol', '')

        parts = filepath.split('/')
        date_folder = ''
        for part in parts:
            if re.match(r'\d{4}-\d{2}', part):
                date_folder = part
                break

        header_lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                header_lines.append(stripped.lstrip('/*').lstrip('/ ').strip())
            elif stripped.startswith('pragma') or stripped.startswith('import') or stripped == '':
                continue
            else:
                break

        description = ' '.join(header_lines).strip()

        vuln_type = ''
        desc_lower = description.lower()
        for keyword, vtype in vuln_keywords.items():
            if re.search(keyword, desc_lower):
                vuln_type = vtype
                break

        return {
            'protocol': protocol,
            'date': date_folder,
            'description': description[:300],
            'vuln_type': vuln_type or 'unknown',
            'code': content,
        }

    exploits = []
    for fpath in tqdm(exp_files, desc='Parsing exploit files'):
        result = parse_exploit_file(fpath)
        if result:
            exploits.append(result)

    log(f"DeFiHackLabs: parsed {len(exploits)} exploits")

    # Build pairs — PRD: "same-protocol, different-vulnerability pairs"
    random.seed(42)

    # Group by date folder (exploits in same month target related protocols)
    by_date = defaultdict(list)
    for exp in exploits:
        if exp['date']:
            by_date[exp['date']].append(exp)

    pairs = []
    for exp in tqdm(exploits, desc='Building DeFiHackLabs pairs'):
        title = f'{exp["protocol"]} exploit ({exp["vuln_type"]})'
        desc = exp['description']
        query = f'HIGH severity: {title}. {desc}'

        hard_neg = ''
        neg_type = 'random'

        # Strategy 1: same time period, different vuln type (closest to "same protocol ecosystem")
        # DeFiHackLabs organizes by month — exploits in the same month often target related protocols
        if exp['date'] and len(by_date[exp['date']]) > 1:
            candidates = [e for e in by_date[exp['date']]
                          if e['protocol'] != exp['protocol']
                          and e['vuln_type'] != exp['vuln_type']]
            if candidates:
                hard_neg = random.choice(candidates)['code']
                neg_type = 'same_period_diff_vuln'

        # Strategy 2: same vuln type, different protocol (confusing negative)
        if not hard_neg:
            same_vuln = [e for e in exploits
                          if e['vuln_type'] == exp['vuln_type']
                          and e['protocol'] != exp['protocol']]
            if same_vuln:
                hard_neg = random.choice(same_vuln)['code']
                neg_type = 'same_vuln_diff_protocol'

        # Strategy 3: different vuln, different protocol (random fallback)
        if not hard_neg:
            diff_vuln = [e for e in exploits
                          if e['vuln_type'] != exp['vuln_type'] and e['protocol'] != exp['protocol']]
            if diff_vuln:
                hard_neg = random.choice(diff_vuln)['code']
                neg_type = 'random'

        pairs.append({
            'query': query,
            'positive': exp['code'],
            'hard_negative': hard_neg,
            'negative_type': neg_type,
            'source': 'DeFiHackLabs',
            'severity': 'HIGH',
            'vuln_type': exp['vuln_type'],
            'quality_tier': 2,
        })

    pairs_df = pd.DataFrame(pairs)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    pairs_df.to_parquet(output_path, index=False)
    vol.commit()
    log(f"DeFiHackLabs: saved {len(pairs_df)} pairs")


# ---------------------------------------------------------------------------
# 08 — msc (vulnerability-severity-classification)
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=3600)
def process_msc():
    """Load msc dataset, merge train+test, build vuln-vs-safe pairs."""
    import random
    import pandas as pd
    from datasets import load_dataset, concatenate_datasets

    output_path = f"{INTERMEDIATE_DIR}/msc_pairs_processed.parquet"
    if os.path.exists(output_path):
        log("MSC: already exists, skipping")
        return

    log("MSC: loading dataset...")
    ds = load_dataset('msc-smart-contract-auditing/vulnerability-severity-classification')
    merged = concatenate_datasets([ds['train'], ds['test']])
    df = merged.to_pandas()
    log(f"MSC: loaded {len(df)} rows (train={len(ds['train'])}, test={len(ds['test'])})")

    random.seed(42)
    vuln_df = df[df['severity'].isin(['high', 'medium'])].copy()
    safe_df = df[df['severity'] == 'none'].copy()
    safe_codes = safe_df['function'].tolist()
    log(f"MSC: {len(vuln_df)} vulnerable, {len(safe_df)} safe functions")
    # PRD: "vulnerable vs safe function pairs from same audit firm"
    # msc dataset has no firm/auditor column — we pair vuln vs ANY safe function.
    # Hard negative is still meaningful: vuln function vs safe function from same dataset.

    pairs = []
    for _, row in vuln_df.iterrows():
        vuln_code = row['function']
        severity = row['severity']

        code_lines = [l.strip() for l in vuln_code.splitlines()
                       if l.strip() and not l.strip().startswith('//')]

        func_sig = ''
        for line in code_lines:
            if 'function ' in line:
                func_sig = line.strip()
                break
        if not func_sig and code_lines:
            func_sig = code_lines[0][:80]

        title = f'Vulnerable function: {func_sig[:100]}'
        desc = vuln_code[:300]
        query = f'{severity.upper()} severity: {title}. {desc}'

        hard_neg = random.choice(safe_codes) if safe_codes else ''

        pairs.append({
            'query': query,
            'positive': vuln_code,
            'hard_negative': hard_neg,
            'negative_type': 'same_firm_safe',
            'source': 'msc-smart-contract-auditing',
            'severity': severity.upper(),
            'vuln_type': '',
            'quality_tier': 3,
        })

    pairs_df = pd.DataFrame(pairs)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    pairs_df.to_parquet(output_path, index=False)
    vol.commit()
    log(f"MSC: saved {len(pairs_df)} pairs")


# ---------------------------------------------------------------------------
# 09 — SmartBugs Curated
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=3600)
def process_smartbugs_curated():
    """Clone smartbugs-curated, extract by DASP taxonomy, build pairs."""
    import subprocess
    import glob
    import random
    import pandas as pd
    from tqdm import tqdm

    output_path = f"{INTERMEDIATE_DIR}/smartbugs_curated_pairs.parquet"
    if os.path.exists(output_path):
        log("SmartBugs-Curated: already exists, skipping")
        return

    log("SmartBugs-Curated: cloning repo...")
    clone_dir = "/tmp/smartbugs-curated"
    subprocess.run(["git", "clone", "--depth", "1",
                     "https://github.com/smartbugs/smartbugs-curated.git", clone_dir],
                    check=True, capture_output=True)

    # Find dataset directory
    dataset_dir = clone_dir
    for candidate in ['dataset', 'contracts', 'vulnerabilities', '.']:
        test_path = os.path.join(clone_dir, candidate)
        if os.path.isdir(test_path):
            sol_check = glob.glob(os.path.join(test_path, '**/*.sol'), recursive=True)
            if sol_check:
                dataset_dir = test_path
                break

    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    log(f"SmartBugs-Curated: found {len(subdirs)} categories in {dataset_dir}")

    random.seed(42)
    all_contracts = []
    for category in sorted(subdirs):
        cat_dir = os.path.join(dataset_dir, category)
        sol_files = glob.glob(os.path.join(cat_dir, '**/*.sol'), recursive=True)
        for fpath in sol_files:
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    source_code = f.read()
            except Exception:
                continue
            if source_code.strip():
                all_contracts.append({
                    'code': source_code,
                    'vuln_type': category,
                    'filename': os.path.basename(fpath),
                })

    log(f"SmartBugs-Curated: found {len(all_contracts)} contracts")

    pairs = []
    for i, contract in enumerate(tqdm(all_contracts, desc='Building SmartBugs-Curated pairs')):
        vuln_type = contract['vuln_type'].replace('_', ' ').replace('-', ' ').title()
        title = f'{vuln_type} vulnerability in {contract["filename"].replace(".sol", "")}'
        desc = contract['code'][:300]
        query = f'HIGH severity: {title}. {desc}'

        diff_category = [c for c in all_contracts if c['vuln_type'] != contract['vuln_type']]
        hard_neg = random.choice(diff_category)['code'] if diff_category else ''

        pairs.append({
            'query': query,
            'positive': contract['code'],
            'hard_negative': hard_neg,
            'negative_type': 'diff_dasp_category',
            'source': 'SmartBugs-Curated',
            'severity': 'HIGH',
            'vuln_type': contract['vuln_type'],
            'quality_tier': 1,
        })

    pairs_df = pd.DataFrame(pairs)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    pairs_df.to_parquet(output_path, index=False)
    vol.commit()
    log(f"SmartBugs-Curated: saved {len(pairs_df)} pairs")


# ═══════════════════════════════════════════════════════════════════════════
# WAVE 2: Merge & Build (3 functions, run sequentially)
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 10 — Merge Stream 1 (SAE corpus)
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol},
              secrets=[HF_SECRET], timeout=7200)
def merge_stream1():
    """Load 5 Stream 1 intermediates, cross-dedup, push to HuggingFace."""
    import pandas as pd
    from datasets import Dataset

    vol.reload()

    output_path = f"{INTERMEDIATE_DIR}/stream1_merged.parquet"
    push_marker = f"{INTERMEDIATE_DIR}/stream1_pushed.marker"

    if os.path.exists(push_marker):
        log("Stream 1 merge: already merged and pushed, skipping")
        return

    # If intermediate exists but push didn't complete, load and retry push
    if os.path.exists(output_path):
        log("Stream 1 merge: intermediate exists, retrying HF push...")
        merged = pd.read_parquet(output_path)
    else:
        merged = _merge_stream1_process(output_path)

    # Push to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    hf_ds = Dataset.from_pandas(merged, preserve_index=False)
    repo_id = f"{HF_USERNAME}/scar-corpus"
    hf_ds.push_to_hub(repo_id, private=True, token=hf_token,
                       commit_message=f'Upload SAE corpus ({len(merged)} contracts)')
    log(f"Stream 1: pushed {len(merged)} rows to {repo_id}")

    # Mark push complete
    with open(push_marker, 'w') as f:
        f.write('done')
    vol.commit()


def _merge_stream1_process(output_path):
    """Internal: run the actual merge processing. Returns merged DataFrame."""
    import pandas as pd

    STREAM1_FILES = {
        'DISL': f'{INTERMEDIATE_DIR}/disl_processed.parquet',
        'slither-audited': f'{INTERMEDIATE_DIR}/slither_audited_processed.parquet',
        'DeFiVulnLabs': f'{INTERMEDIATE_DIR}/defivulnlabs_processed.parquet',
        'FORGE-Curated': f'{INTERMEDIATE_DIR}/forge_stream1_contracts.parquet',
        'SmartBugs-Wild': f'{INTERMEDIATE_DIR}/smartbugs_wild_processed.parquet',
    }

    EXPECTED_COLS = ['source', 'contract_code', 'num_lines', 'has_vuln_labels',
                     'vuln_labels', 'contract_hash']

    dfs = []
    for name, path in STREAM1_FILES.items():
        if not os.path.exists(path):
            log(f"Stream 1: WARNING — missing {name} at {path}")
            continue
        df = pd.read_parquet(path)
        log(f"  {name}: {len(df)} rows")

        for col in EXPECTED_COLS:
            if col not in df.columns:
                if col == 'has_vuln_labels':
                    df[col] = False
                elif col == 'vuln_labels':
                    df[col] = ''
                else:
                    raise ValueError(f'{name} missing required column: {col}')

        df['vuln_labels'] = df['vuln_labels'].apply(
            lambda x: '|'.join(x) if isinstance(x, (list, tuple)) else str(x) if x else ''
        )
        df = df[EXPECTED_COLS]
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    log(f"Stream 1: {len(merged)} total before dedup")

    # Cross-source dedup (prefer labeled contracts)
    merged['_sort_key'] = merged['has_vuln_labels'].apply(lambda x: 0 if x else 1)
    merged = merged.sort_values('_sort_key').drop(columns='_sort_key')
    merged = merged.drop_duplicates(subset='contract_hash', keep='first').reset_index(drop=True)
    log(f"Stream 1: {len(merged)} after cross-source dedup")

    # Final quality pass — recalculate line count
    merged['num_lines'] = merged['contract_code'].apply(count_lines)
    # DeFiVulnLabs exempted from 50-line filter: vulnerability labs are intentionally
    # short (~20-40 lines) to isolate specific vuln patterns for SAE feature learning
    mask_short = (merged['num_lines'] < 50) & (merged['source'] != 'DeFiVulnLabs')
    merged = merged[~mask_short].reset_index(drop=True)
    merged = merged[merged['contract_code'].str.strip().str.len() > 0].reset_index(drop=True)
    log(f"Stream 1: {len(merged)} after quality pass")

    # Safety net: remove any contracts whose hash matches Stream 3 eval ground truth
    # FORGE contracts dir is not split by report, so some eval-report contracts may leak.
    # SAE pretraining is unsupervised so risk is low, but we enforce zero overlap anyway.
    stream3_path = f"{INTERMEDIATE_DIR}/forge_stream3_eval.parquet"
    if os.path.exists(stream3_path):
        s3_df = pd.read_parquet(stream3_path)
        gt_col = 'ground_truth_code' if 'ground_truth_code' in s3_df.columns else 'positive'
        s3_hashes = set()
        for code in s3_df[gt_col]:
            if code and str(code).strip():
                s3_hashes.add(hash_contract(str(code)))
        before_s3_check = len(merged)
        merged = merged[~merged['contract_hash'].isin(s3_hashes)].reset_index(drop=True)
        removed = before_s3_check - len(merged)
        if removed > 0:
            log(f"Stream 1: removed {removed} contracts matching Stream 3 eval hashes")
        else:
            log("Stream 1: no overlap with Stream 3 eval (clean)")

    # Save intermediate
    merged.to_parquet(output_path, index=False)
    vol.commit()

    return merged


# ---------------------------------------------------------------------------
# 11 — Build Stream 2 (contrastive pairs)
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol},
              secrets=[HF_SECRET], timeout=7200)
def build_stream2():
    """Load 5 Stream 2 intermediates, fill hard negs, leakage check, push to HF."""
    import random
    import pandas as pd
    from datasets import Dataset
    from tqdm import tqdm

    vol.reload()

    output_path = f"{INTERMEDIATE_DIR}/stream2_merged.parquet"
    push_marker = f"{INTERMEDIATE_DIR}/stream2_pushed.marker"

    if os.path.exists(push_marker):
        log("Stream 2 build: already merged and pushed, skipping")
        return

    # If intermediate exists but push didn't complete, load and retry push
    if os.path.exists(output_path):
        log("Stream 2 build: intermediate exists, retrying HF push...")
        merged = pd.read_parquet(output_path)
    else:
        merged = _build_stream2_process(output_path)

    # Push to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    hf_ds = Dataset.from_pandas(merged, preserve_index=False)
    repo_id = f"{HF_USERNAME}/scar-pairs"
    hf_ds.push_to_hub(repo_id, private=True, token=hf_token,
                       commit_message=f'Upload contrastive pairs ({len(merged)} pairs)')
    log(f"Stream 2: pushed {len(merged)} pairs to {repo_id}")

    # Mark push complete
    with open(push_marker, 'w') as f:
        f.write('done')
    vol.commit()


def _build_stream2_process(output_path):
    """Internal: run the actual Stream 2 build processing. Returns merged DataFrame."""
    import random
    import pandas as pd
    from tqdm import tqdm

    random.seed(42)

    STREAM2_FILES = {
        'Solodit': f'{INTERMEDIATE_DIR}/solodit_pairs.parquet',
        'DeFiHackLabs': f'{INTERMEDIATE_DIR}/defihacklabs_pairs.parquet',
        'MSC': f'{INTERMEDIATE_DIR}/msc_pairs_processed.parquet',
        'FORGE-Curated': f'{INTERMEDIATE_DIR}/forge_stream2_pairs.parquet',
        'SmartBugs-Curated': f'{INTERMEDIATE_DIR}/smartbugs_curated_pairs.parquet',
    }

    EXPECTED_COLS = ['query', 'positive', 'hard_negative', 'negative_type',
                     'source', 'severity', 'vuln_type', 'quality_tier']

    dfs = []
    for name, path in STREAM2_FILES.items():
        if not os.path.exists(path):
            log(f"Stream 2: WARNING — missing {name} at {path}")
            continue
        df = pd.read_parquet(path)
        log(f"  {name}: {len(df)} pairs")

        for col in EXPECTED_COLS:
            if col not in df.columns:
                if col == 'vuln_type':
                    df[col] = ''
                elif col == 'quality_tier':
                    df[col] = 3
                elif col == 'negative_type':
                    df[col] = 'random'
                elif col == 'hard_negative':
                    df[col] = ''
                else:
                    raise ValueError(f'{name} missing required column: {col}')

        df['quality_tier'] = df['quality_tier'].astype(int)
        df['severity'] = df['severity'].str.upper()
        df['vuln_type'] = df['vuln_type'].fillna('')
        df['hard_negative'] = df['hard_negative'].fillna('')
        df = df[EXPECTED_COLS]
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    log(f"Stream 2: {len(merged)} total pairs")

    # Fill missing hard negatives
    random.seed(42)
    empty_neg_mask = merged['hard_negative'].str.strip().str.len() == 0
    n_empty = empty_neg_mask.sum()

    if n_empty > 0:
        all_positives = merged['positive'].tolist()
        for idx in merged[empty_neg_mask].index:
            current_positive = merged.at[idx, 'positive']
            candidates = [p for p in all_positives if p != current_positive]
            if candidates:
                merged.at[idx, 'hard_negative'] = random.choice(candidates)
                merged.at[idx, 'negative_type'] = 'cross_source_random'
        remaining = (merged['hard_negative'].str.strip().str.len() == 0).sum()
        if remaining > 0:
            merged = merged[merged['hard_negative'].str.strip().str.len() > 0].reset_index(drop=True)
        filled = (n_empty - remaining) if remaining else n_empty
        log(f"Stream 2: filled {filled} empty hard negatives")

    # Leakage check against Stream 3
    stream3_path = f"{INTERMEDIATE_DIR}/forge_stream3_eval.parquet"
    if os.path.exists(stream3_path):
        eval_df = pd.read_parquet(stream3_path)
        eval_code_col = 'ground_truth_code' if 'ground_truth_code' in eval_df.columns else 'positive'
        eval_hashes = set()
        for code in eval_df[eval_code_col]:
            if code and str(code).strip():
                eval_hashes.add(hash_contract(str(code)))

        leaked_indices = []
        for idx, row in merged.iterrows():
            pos_hash = hash_contract(str(row['positive']))
            if pos_hash in eval_hashes:
                leaked_indices.append(idx)

        if leaked_indices:
            log(f"Stream 2: REMOVING {len(leaked_indices)} pairs leaked from Stream 3")
            merged = merged.drop(leaked_indices).reset_index(drop=True)

    # Dedup by positive hash
    merged['_pos_hash'] = merged['positive'].apply(hash_contract)
    before = len(merged)
    merged = merged.drop_duplicates(subset='_pos_hash', keep='first')
    merged = merged.drop(columns='_pos_hash').reset_index(drop=True)
    log(f"Stream 2: {before} → {len(merged)} after dedup")

    # Save intermediate
    merged.to_parquet(output_path, index=False)
    vol.commit()

    return merged


# ---------------------------------------------------------------------------
# 12 — Build Stream 3 (eval set)
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={INTERMEDIATE_DIR: vol},
              secrets=[HF_SECRET], timeout=3600)
def build_stream3():
    """Load FORGE eval holdout, leakage check vs S1/S2, push to HF."""
    import json
    import pandas as pd
    from datasets import Dataset

    vol.reload()

    output_path = f"{INTERMEDIATE_DIR}/stream3_final.parquet"
    push_marker = f"{INTERMEDIATE_DIR}/stream3_pushed.marker"

    if os.path.exists(push_marker):
        log("Stream 3 build: already built and pushed, skipping")
        return

    # If intermediate exists but push didn't complete, load and retry push
    if os.path.exists(output_path):
        log("Stream 3 build: intermediate exists, retrying HF push...")
        eval_df = pd.read_parquet(output_path)
    else:
        eval_df = _build_stream3_process(output_path)

    # Push to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    hf_ds = Dataset.from_pandas(eval_df, preserve_index=False)
    repo_id = f"{HF_USERNAME}/scar-eval"
    hf_ds.push_to_hub(repo_id, private=True, token=hf_token,
                       commit_message=f'Upload eval set ({len(eval_df)} findings, zero-leakage)')
    log(f"Stream 3: pushed {len(eval_df)} findings to {repo_id}")

    # Mark push complete
    with open(push_marker, 'w') as f:
        f.write('done')
    vol.commit()


def _build_stream3_process(output_path):
    """Internal: run the actual Stream 3 build processing. Returns eval DataFrame."""
    import json
    import pandas as pd

    eval_df = pd.read_parquet(f"{INTERMEDIATE_DIR}/forge_stream3_eval.parquet")
    log(f"Stream 3: loaded {len(eval_df)} eval findings")

    EXPECTED_COLS = ['query', 'ground_truth_code', 'severity', 'vuln_type',
                     'report_name', 'audit_firm', 'report_date']

    # Map column names if needed
    if 'positive' in eval_df.columns and 'ground_truth_code' not in eval_df.columns:
        eval_df = eval_df.rename(columns={'positive': 'ground_truth_code'})

    for col in EXPECTED_COLS:
        if col not in eval_df.columns:
            eval_df[col] = ''

    eval_df = eval_df[[col for col in EXPECTED_COLS if col in eval_df.columns]]
    eval_df['severity'] = eval_df['severity'].str.upper().replace('CRITICAL', 'HIGH')
    eval_df = eval_df.fillna('')

    # Remove empty entries
    eval_df = eval_df[eval_df['ground_truth_code'].str.strip().str.len() > 0].reset_index(drop=True)
    eval_df = eval_df[eval_df['query'].str.strip().str.len() > 0].reset_index(drop=True)
    log(f"Stream 3: {len(eval_df)} after quality filter")

    # Leakage check vs Stream 1
    eval_hashes = set()
    for code in eval_df['ground_truth_code']:
        if code and str(code).strip():
            eval_hashes.add(hash_contract(str(code)))

    s1_path = f"{INTERMEDIATE_DIR}/stream1_merged.parquet"
    if os.path.exists(s1_path):
        s1 = pd.read_parquet(s1_path, columns=['contract_hash'])
        s1_hashes = set(s1['contract_hash'].tolist())
        overlap = eval_hashes & s1_hashes
        if overlap:
            log(f"Stream 3: WARNING — {len(overlap)} hashes found in Stream 1!")

    # Leakage check vs Stream 2
    s2_path = f"{INTERMEDIATE_DIR}/stream2_merged.parquet"
    if os.path.exists(s2_path):
        s2 = pd.read_parquet(s2_path, columns=['positive'])
        s2_hashes = set()
        for code in s2['positive']:
            if code and str(code).strip():
                s2_hashes.add(hash_contract(str(code)))
        overlap = eval_hashes & s2_hashes
        if overlap:
            log(f"Stream 3: WARNING — {len(overlap)} hashes found in Stream 2!")

    # Verify split metadata
    meta_path = f"{INTERMEDIATE_DIR}/forge_split_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        s1_reports = set(meta.get('stream1_reports', []))
        s2_reports = set(meta.get('stream2_reports', []))
        s3_reports = set(meta.get('stream3_reports', []))
        assert not (s1_reports & s3_reports), 'LEAK: Reports in both Stream 1 and Stream 3!'
        assert not (s2_reports & s3_reports), 'LEAK: Reports in both Stream 2 and Stream 3!'
        log("Stream 3: report-level isolation verified")

    # Save intermediate
    eval_df.to_parquet(output_path, index=False)
    vol.commit()

    return eval_df


# ═══════════════════════════════════════════════════════════════════════════
# WAVE 3: Validation
# ═══════════════════════════════════════════════════════════════════════════

@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=3600)
def validate():
    """Run all validation checks on the 3 merged datasets."""
    import json
    import pandas as pd
    from collections import Counter
    from datetime import datetime

    vol.reload()

    log("=" * 60)
    log("VALIDATION STARTING")
    log("=" * 60)

    results = {}

    # Load datasets
    s1 = pd.read_parquet(f"{INTERMEDIATE_DIR}/stream1_merged.parquet")
    s2 = pd.read_parquet(f"{INTERMEDIATE_DIR}/stream2_merged.parquet")

    s3_path = f"{INTERMEDIATE_DIR}/stream3_final.parquet"
    if not os.path.exists(s3_path):
        s3_path = f"{INTERMEDIATE_DIR}/forge_stream3_eval.parquet"
    s3 = pd.read_parquet(s3_path)

    results['s1_count'] = len(s1)
    results['s2_count'] = len(s2)
    results['s3_count'] = len(s3)
    log(f"Loaded: S1={len(s1)}, S2={len(s2)}, S3={len(s3)}")

    # --- CHECK 1: Size thresholds ---
    MIN_S1, MIN_S2, MIN_S3 = 200000, 20000, 400
    size_checks = [
        ('Stream 1 >= 200k', len(s1) >= MIN_S1, f'{len(s1):,}'),
        ('Stream 2 >= 20k', len(s2) >= MIN_S2, f'{len(s2):,}'),
        ('Stream 3 >= 400', len(s3) >= MIN_S3, f'{len(s3):,}'),
    ]
    results['size_check'] = 'PASS' if all(c[1] for c in size_checks) else 'WARN'
    for name, passed, val in size_checks:
        log(f"  {name}: {'PASS' if passed else 'WARN'} ({val})")

    # --- CHECK 2: Dedup verification ---
    s1_dedup = s1['contract_hash'].nunique() == len(s1)
    s2['_pos_hash'] = s2['positive'].apply(hash_contract)
    s2_dedup = s2['_pos_hash'].nunique() == len(s2)

    gt_col = 'ground_truth_code' if 'ground_truth_code' in s3.columns else 'positive'
    s3['_gt_hash'] = s3[gt_col].apply(hash_contract)
    s3_dedup = s3['_gt_hash'].nunique() == len(s3)

    results['dedup_s1'] = 'PASS' if s1_dedup else 'WARN'
    results['dedup_s2'] = 'PASS' if s2_dedup else 'WARN'
    results['dedup_s3'] = 'PASS' if s3_dedup else 'WARN'
    log(f"  Dedup S1: {results['dedup_s1']}, S2: {results['dedup_s2']}, S3: {results['dedup_s3']}")

    # --- CHECK 3: Leakage S3 vs S1 ---
    eval_hashes = set(s3['_gt_hash'].tolist())
    s1_hashes = set(s1['contract_hash'].tolist())
    overlap_s1 = eval_hashes & s1_hashes
    results['leakage_s1_s3'] = 'PASS' if not overlap_s1 else f'FAIL ({len(overlap_s1)})'
    log(f"  Leakage S3↔S1: {results['leakage_s1_s3']}")

    # --- CHECK 4: Leakage S3 vs S2 ---
    s2_pos_hashes = set(s2['_pos_hash'].tolist())
    overlap_s2 = eval_hashes & s2_pos_hashes
    results['leakage_s2_s3'] = 'PASS' if not overlap_s2 else f'FAIL ({len(overlap_s2)})'

    s2_neg_hashes = set()
    for neg in s2['hard_negative']:
        if neg and str(neg).strip():
            s2_neg_hashes.add(hash_contract(str(neg)))
    overlap_neg = eval_hashes & s2_neg_hashes
    results['leakage_s2neg_s3'] = 'PASS' if not overlap_neg else f'WARN ({len(overlap_neg)})'
    log(f"  Leakage S3↔S2pos: {results['leakage_s2_s3']}, S3↔S2neg: {results['leakage_s2neg_s3']}")

    # --- CHECK 5: Report isolation ---
    meta_path = f"{INTERMEDIATE_DIR}/forge_split_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        s1r = set(meta.get('stream1_reports', []))
        s2r = set(meta.get('stream2_reports', []))
        s3r = set(meta.get('stream3_reports', []))
        results['report_isolation'] = 'PASS' if (not (s1r & s3r) and not (s2r & s3r)) else 'FAIL'
    else:
        results['report_isolation'] = 'SKIP (no metadata)'
    log(f"  Report isolation: {results['report_isolation']}")

    # --- CHECK 6-8: Distribution stats ---
    log("\n--- Stream 1 Distribution ---")
    s1_sources = s1['source'].value_counts()
    for src, cnt in s1_sources.items():
        log(f"  {src}: {cnt:,}")

    log("\n--- Stream 2 Distribution ---")
    s2_sources = s2['source'].value_counts()
    for src, cnt in s2_sources.items():
        log(f"  {src}: {cnt:,}")

    sev_counts = s2['severity'].value_counts()
    log(f"  Severity: {dict(sev_counts)}")
    neg_counts = s2['negative_type'].value_counts()
    log(f"  Neg types: {dict(neg_counts)}")

    log("\n--- Stream 3 Distribution ---")
    s3_sev = s3['severity'].value_counts()
    log(f"  Severity: {dict(s3_sev)}")
    if 'report_name' in s3.columns:
        log(f"  Unique reports: {s3['report_name'].nunique()}")

    # --- CHECK 9: Vulnerability coverage ---
    KNOWN_VULN_TYPES = [
        'reentrancy', 'access-control', 'arithmetic', 'unchecked-calls',
        'denial-of-service', 'bad-randomness', 'front-running', 'time-manipulation',
        'short-addresses', 'flash-loan', 'price-manipulation', 'oracle-manipulation',
        'sandwich-attack', 'governance', 'token-approval', 'integer-overflow',
        'integer-underflow', 'uninitialized-storage', 'delegatecall',
        'selfdestruct', 'tx-origin', 'signature-replay', 'cross-chain',
    ]

    all_labels = set()
    s1_labeled = s1[s1['has_vuln_labels']]
    for labels_str in s1_labeled['vuln_labels']:
        if labels_str:
            for label in str(labels_str).split('|'):
                if label.strip():
                    all_labels.add(label.strip().lower())

    for vt in s2['vuln_type']:
        if vt and str(vt).strip():
            all_labels.add(str(vt).strip().lower())

    if 'vuln_type' in s3.columns:
        for vt in s3['vuln_type']:
            if vt and str(vt).strip():
                all_labels.add(str(vt).strip().lower())

    covered = [vt for vt in KNOWN_VULN_TYPES if any(vt in label for label in all_labels)]
    results['vuln_coverage'] = f'{len(covered)}/{len(KNOWN_VULN_TYPES)}'
    log(f"\n  Vuln coverage: {results['vuln_coverage']}")
    log(f"  Covered: {covered}")
    log(f"  Missing: {[vt for vt in KNOWN_VULN_TYPES if vt not in covered]}")

    # --- CHECK 10: Code quality ---
    solidity_keywords = ['function', 'contract', 'pragma', 'mapping', 'require',
                          'modifier', 'event', 'emit', 'address', 'uint256',
                          'bytes32', 'msg.sender']

    def has_sol_markers(code: str) -> bool:
        code_lower = str(code).lower()
        return sum(1 for kw in solidity_keywords if kw in code_lower) >= 2

    s2_sol_pct = s2['positive'].apply(has_sol_markers).mean() * 100
    s3_sol_pct = s3[gt_col].apply(has_sol_markers).mean() * 100
    s2_q_pct = s2['query'].str.match(r'^(HIGH|MEDIUM|CRITICAL) severity:').mean() * 100

    results['code_quality_s2'] = f'{s2_sol_pct:.1f}%'
    results['code_quality_s3'] = f'{s3_sol_pct:.1f}%'
    results['query_format_s2'] = f'{s2_q_pct:.1f}%'
    log(f"\n  Code quality — S2 positives: {s2_sol_pct:.1f}% Solidity")
    log(f"  Code quality — S3 ground truth: {s3_sol_pct:.1f}% Solidity")
    log(f"  Query format compliance: {s2_q_pct:.1f}%")

    # --- FINAL DASHBOARD ---
    critical_checks = ['leakage_s1_s3', 'leakage_s2_s3', 'report_isolation']
    critical_pass = all(results.get(c, '') == 'PASS' for c in critical_checks)

    log("\n" + "=" * 60)
    if critical_pass:
        log("ALL CRITICAL CHECKS PASSED")
    else:
        failed = [c for c in critical_checks if results.get(c, '') != 'PASS']
        log(f"CRITICAL CHECKS FAILED: {failed}")
    log("=" * 60)

    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'stream1_count': int(results['s1_count']),
        'stream2_count': int(results['s2_count']),
        'stream3_count': int(results['s3_count']),
        'stream1_sources': s1['source'].value_counts().to_dict(),
        'stream2_sources': s2['source'].value_counts().to_dict(),
        'stream2_severity': s2['severity'].value_counts().to_dict(),
        'stream2_neg_types': s2['negative_type'].value_counts().to_dict(),
        'checks': results,
        'critical_passed': critical_pass,
    }

    report_path = f"{INTERMEDIATE_DIR}/validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    vol.commit()
    log(f"Validation report saved to {report_path}")

    # Clean up temp columns
    s2.drop(columns=['_pos_hash'], inplace=True, errors='ignore')
    s3.drop(columns=['_gt_hash'], inplace=True, errors='ignore')


# ═══════════════════════════════════════════════════════════════════════════
# FORCE-RERUN HELPER
# ═══════════════════════════════════════════════════════════════════════════

@app.function(image=image, volumes={INTERMEDIATE_DIR: vol}, timeout=300)
def _delete_for_rerun():
    """Delete Solodit + Stream 2 intermediates to force reprocessing."""
    vol.reload()
    targets = [
        f"{INTERMEDIATE_DIR}/solodit_pairs.parquet",
        f"{INTERMEDIATE_DIR}/stream2_merged.parquet",
        f"{INTERMEDIATE_DIR}/stream2_pushed.marker",
    ]
    for path in targets:
        if os.path.exists(path):
            os.remove(path)
            log(f"Deleted: {path}")
        else:
            log(f"Not found (ok): {path}")
    vol.commit()
    log("Old intermediates deleted — Solodit + Stream 2 will reprocess")


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(wave: int = 0, force_solodit: bool = False):
    """Run the full pipeline or a specific wave.

    Args:
        wave: 0=all (default), 1=source processing, 2=merge, 3=validate
        force_solodit: if True, delete old Solodit + Stream 2 intermediates to force reprocessing
    """
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] SCAR Pipeline starting (wave={wave})")

    if force_solodit:
        print("--- FORCE: deleting old Solodit + Stream 2 intermediates ---")
        _delete_for_rerun.remote()

    if wave == 0 or wave == 1:
        print("--- WAVE 1: Source Processing (9 parallel containers) ---")
        handles = [
            process_disl.spawn(),
            process_slither.spawn(),
            process_defivulnlabs.spawn(),
            process_forge.spawn(),
            process_smartbugs_wild.spawn(),
            process_solodit.spawn(),
            process_defihacklabs.spawn(),
            process_msc.spawn(),
            process_smartbugs_curated.spawn(),
        ]
        for i, h in enumerate(handles):
            result = h.get()
            print(f"  Wave 1 function {i+1}/9 completed")
        print("--- WAVE 1 COMPLETE ---\n")

    if wave == 0 or wave == 2:
        print("--- WAVE 2: Merge & Build (sequential) ---")
        merge_stream1.remote()
        print("  Stream 1 merge complete")
        build_stream2.remote()
        print("  Stream 2 build complete")
        build_stream3.remote()
        print("  Stream 3 build complete")
        print("--- WAVE 2 COMPLETE ---\n")

    if wave == 0 or wave == 3:
        print("--- WAVE 3: Validation ---")
        validate.remote()
        print("--- WAVE 3 COMPLETE ---\n")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Pipeline finished!")
    print("Check your HuggingFace account for:")
    print(f"  - {HF_USERNAME}/scar-corpus")
    print(f"  - {HF_USERNAME}/scar-pairs")
    print(f"  - {HF_USERNAME}/scar-eval")
