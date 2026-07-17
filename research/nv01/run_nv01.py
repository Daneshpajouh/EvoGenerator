#!/usr/bin/env python3
"""Frozen NV-01 public-data execution with two independent code paths."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from scipy.stats import beta

ALPHA = 0.05
PREREG_SHA256 = "b3a35b3d9efaa9b479dd118c09b40ee65138f13784eaa2c0d17aa22ab8bf8ffb"
URLS = [
    "https://ftp.ebi.ac.uk/pub/databases/impc/all-data-releases/release-15.0/results/viability.csv.gz",
    "http://ftp.ebi.ac.uk/pub/databases/impc/all-data-releases/release-15.0/results/viability.csv.gz",
]
OUT = Path("research/nv01/results")


def download() -> tuple[bytes, str, list[dict]]:
    errors: list[dict] = []
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NV01-research-audit/1.0"})
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.read(), url, errors
        except Exception as exc:
            errors.append({"url": url, "error": repr(exc)})
    raise RuntimeError(json.dumps(errors, sort_keys=True))


# --------------------------- independent path A ---------------------------
def a_norm(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def a_first(columns: list[str], exact: list[str], tokens: list[str]) -> str | None:
    mapping = {a_norm(c): c for c in columns}
    for candidate in exact:
        if a_norm(candidate) in mapping:
            return mapping[a_norm(candidate)]
    for column in columns:
        normalized = a_norm(column)
        if any(token in normalized for token in tokens):
            return column
    return None


def a_columns(columns: list[str], tokens: list[str]) -> list[str]:
    return [c for c in columns if any(token in a_norm(c) for token in tokens)]


def a_label(value: str) -> str:
    text = a_norm(value)
    if not text:
        return "unresolved"
    if "subviable" in text or "sub viable" in text or "lethal" in text:
        return "adverse"
    if text == "viable" or text.startswith("viable ") or text.endswith(" viable"):
        return "viable"
    return "unresolved"


def a_cp_upper(x: int, n: int) -> float | None:
    if n <= 0:
        return None
    if x >= n:
        return 1.0
    return float(beta.ppf(1.0 - ALPHA, x + 1, n - x))


def path_a(raw: bytes, source_url: str) -> dict:
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text)
        columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    gene = a_first(columns, ["Gene Accession Id", "Gene Accession ID", "MGI Accession ID", "mgi_id"], ["gene accession", "mgi accession"])
    label = a_first(columns, ["Viability Phenotype HOMs/HEMIs", "Viability Phenotype", "viability_impc"], ["viability phenotype"])
    comment = a_first(columns, ["Comment", "Comments"], ["comment"])
    allele = a_columns(columns, ["allele accession", "allele id", "allele symbol", "allele name"])
    centres = a_columns(columns, ["phenotyping center", "phenotyping centre", "production center", "production centre"])
    lines = a_columns(columns, ["colony id", "line id", "mouse line", "colony name"])
    schema = {"columns": columns, "gene_column": gene, "label_column": label, "comment_column": comment, "allele_columns": allele, "centre_columns": centres, "line_columns": lines}
    result = {"analysis_id": "NV-01", "code_path": "A_csv_scipy", "execution_state": "EXECUTED", "source": {"url": source_url, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}, "schema": schema, "raw_row_count": len(rows)}
    if not gene or not label:
        return result | {"decision": "BLOCKED", "blocker": "Required gene or viability-label column is absent."}

    identity = allele or centres or lines
    basis = "allele" if allele else "centre" if centres else "line" if lines else None
    result["independence_basis"] = basis
    result["independence_columns"] = identity

    counts: Counter[str] = Counter()
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        payload = "\x1f".join((row.get(c) or "").strip() for c in columns)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        counts[digest] += 1
        unique.setdefault(digest, row)
    result["exact_duplicate_row_count"] = sum(v - 1 for v in counts.values())
    result["unique_row_count"] = len(unique)
    if not identity:
        return result | {"decision": "BLOCKED", "blocker": "No explicit allele, centre, colony, or line field supports independent assessments.", "fastest_kill_triggered": "explicit independence unavailable"}

    buckets: dict[tuple[str, tuple[str, ...]], dict] = {}
    for digest in sorted(unique):
        row = unique[digest]
        gene_id = (row.get(gene) or "").strip()
        if not gene_id:
            continue
        ident = tuple((row.get(c) or "").strip() for c in identity)
        missing = not any(ident)
        if missing:
            ident = ("__MISSING_INDEPENDENCE__",)
        key = (gene_id, ident)
        item = buckets.setdefault(key, {"labels": set(), "hashes": [], "centres": set()})
        state = a_label(row.get(label) or "")
        if missing or (comment and (row.get(comment) or "").strip()):
            state = "unresolved"
        item["labels"].add(state)
        item["hashes"].append(digest)
        for c in centres:
            value = (row.get(c) or "").strip()
            if value:
                item["centres"].add(value)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for (gene_id, ident), item in buckets.items():
        clean = item["labels"] - {"unresolved"}
        state = next(iter(clean)) if len(clean) == 1 and "unresolved" not in item["labels"] else "unresolved"
        grouped[gene_id].append({"id": ident, "state": state, "hashes": sorted(item["hashes"]), "centres": sorted(item["centres"])})

    n = x = u = eligible = 0
    adverse: list[str] = []
    unresolved: list[str] = []
    gene_centres: dict[str, set[str]] = {}
    for gene_id in sorted(grouped):
        assays = sorted(grouped[gene_id], key=lambda z: (z["id"], z["hashes"]))
        viable_idx = next((i for i, z in enumerate(assays) if z["state"] == "viable"), None)
        if viable_idx is None:
            continue
        rest = [z for i, z in enumerate(assays) if i != viable_idx]
        if not rest:
            continue
        eligible += 1
        resolved = [z for z in rest if z["state"] in {"viable", "adverse"}]
        unknown = [z for z in rest if z["state"] == "unresolved"]
        if resolved:
            n += 1
            if any(z["state"] == "adverse" for z in resolved):
                x += 1
                adverse.append(gene_id)
            gene_centres[gene_id] = {c for z in [assays[viable_idx], *resolved] for c in z["centres"]}
        elif unknown:
            u += 1
            unresolved.append(gene_id)

    p_hat = x / n if n else None
    upper = a_cp_upper(x, n)
    finite = upper - p_hat if upper is not None and p_hat is not None else None
    residual = u / (n + u) if n + u else None
    total = min(1.0, upper + residual) if upper is not None and residual is not None else None
    adverse_set = set(adverse)
    all_centres = sorted({c for values in gene_centres.values() for c in values})
    loo = []
    if centres and n:
        for centre in all_centres:
            kept = [g for g, values in gene_centres.items() if centre not in values]
            nn = len(kept)
            xx = sum(g in adverse_set for g in kept)
            loo.append({"left_out_centre": centre, "N": nn, "X": xx, "B95": a_cp_upper(xx, nn)})
    loo_available = len(all_centres) >= 2 and len(loo) >= 2
    tests = {"independence_supported": basis is not None, "n_ge_59": n >= 59, "b95_le_0_05": upper is not None and upper <= 0.05, "residual_le_sampling": residual is not None and finite is not None and residual <= finite, "total_upper_le_0_10": total is not None and total <= 0.10, "centre_leave_one_out_available": loo_available, "centre_loo_all_n_ge_59": loo_available and all(z["N"] >= 59 for z in loo), "centre_loo_all_b95_le_0_05": loo_available and all(z["B95"] is not None and z["B95"] <= 0.05 for z in loo)}
    if not loo_available:
        decision, blocker = "BLOCKED", "Mandatory centre leave-one-out kill test cannot be run from source fields."
    elif all(tests.values()):
        decision, blocker = "SURVIVOR", None
    else:
        decision, blocker = ("FAILED" if n else "INCONCLUSIVE"), "One or more frozen fastest-kill or decision criteria failed."
    return result | {"eligible_gene_count": eligible, "N": n, "X": x, "U": u, "p_hat": p_hat, "B95": upper, "sampling_term": finite, "residual_R": residual, "two_term_upper_T95": total, "decision_tests": tests, "decision": decision, "blocker": blocker, "adverse_genes": adverse, "unresolved_genes": unresolved, "leave_one_centre_out": loo, "safety_verdict": "PRECLINICAL_ONLY_NO_HUMAN_OR_TREATMENT_CLAIM"}


# --------------------------- independent path B ---------------------------
def b_canon(value: str) -> str:
    return " ".join("".join(ch if ch.isalnum() else " " for ch in value.lower().strip()).split())


def b_pick(headers: list[str], preferred: list[str], groups: list[tuple[str, ...]]) -> str | None:
    mapping = {b_canon(h): h for h in headers}
    for item in preferred:
        if b_canon(item) in mapping:
            return mapping[b_canon(item)]
    for header in headers:
        normalized = b_canon(header)
        if any(all(part in normalized for part in group) for group in groups):
            return header
    return None


def b_all(headers: list[str], groups: list[tuple[str, ...]]) -> list[str]:
    return [h for h in headers if any(all(part in b_canon(h) for part in group) for group in groups)]


def b_state(value: str) -> str:
    text = b_canon(value)
    words = set(text.split())
    if not text:
        return "unknown"
    if "subviable" in words or ("sub" in words and "viable" in words) or "lethal" in words:
        return "bad"
    if text == "viable" or ("viable" in words and "subviable" not in words and "lethal" not in words):
        return "good"
    return "unknown"


def b_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    q = 1.0 - p
    term = q ** n
    total = term
    for i in range(k):
        term *= ((n - i) / (i + 1)) * (p / q)
        total += term
    return min(1.0, max(0.0, total))


def b_upper(x: int, n: int) -> float | None:
    if n < 1:
        return None
    if x >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if b_cdf(x, n, mid) > ALPHA:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def path_b(raw: bytes, source_url: str) -> dict:
    decoded = gzip.decompress(raw).decode("utf-8-sig")
    table = csv.DictReader(io.StringIO(decoded))
    headers = list(table.fieldnames or [])
    records = list(table)
    gene = b_pick(headers, ["Gene Accession Id", "MGI Accession ID", "mgi_id"], [("gene", "accession"), ("mgi", "accession")])
    label = b_pick(headers, ["Viability Phenotype HOMs/HEMIs", "Viability Phenotype"], [("viability", "phenotype")])
    comment = b_pick(headers, ["Comment", "Comments"], [("comment",)])
    allele = b_all(headers, [("allele", "accession"), ("allele", "id"), ("allele", "symbol"), ("allele", "name")])
    centres = b_all(headers, [("phenotyping", "center"), ("phenotyping", "centre"), ("production", "center"), ("production", "centre")])
    lines = b_all(headers, [("colony", "id"), ("line", "id"), ("mouse", "line"), ("colony", "name")])
    schema = {"columns": headers, "gene_column": gene, "label_column": label, "comment_column": comment, "allele_columns": allele, "centre_columns": centres, "line_columns": lines}
    result = {"analysis_id": "NV-01", "code_path": "B_independent_binomial_root", "execution_state": "REPRODUCED", "source": {"url": source_url, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}, "schema": schema, "raw_row_count": len(records)}
    if gene is None or label is None:
        return result | {"decision": "BLOCKED", "blocker": "Required gene or viability-label column is absent."}
    if allele:
        identity, basis = allele, "allele"
    elif centres:
        identity, basis = centres, "centre"
    elif lines:
        identity, basis = lines, "line"
    else:
        identity, basis = [], None
    result["independence_basis"] = basis
    result["independence_columns"] = identity
    unique: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    for row in records:
        payload = b"\x1e".join((row.get(h, "").strip()).encode() for h in headers)
        digest = hashlib.sha256(payload).hexdigest()
        counts[digest] = counts.get(digest, 0) + 1
        unique.setdefault(digest, row)
    result["exact_duplicate_row_count"] = sum(v - 1 for v in counts.values())
    result["unique_row_count"] = len(unique)
    if not identity:
        return result | {"decision": "BLOCKED", "blocker": "No explicit allele, centre, colony, or line field supports independent assessments.", "fastest_kill_triggered": "explicit independence unavailable"}

    buckets: dict[tuple[str, tuple[str, ...]], dict] = {}
    for digest in sorted(unique):
        row = unique[digest]
        gene_id = (row.get(gene) or "").strip()
        if not gene_id:
            continue
        ident = tuple((row.get(h) or "").strip() for h in identity)
        missing = not any(ident)
        if missing:
            ident = ("__MISSING_INDEPENDENCE__",)
        item = buckets.setdefault((gene_id, ident), {"states": set(), "hashes": [], "centres": set()})
        state = b_state(row.get(label) or "")
        if missing or (comment and (row.get(comment) or "").strip()):
            state = "unknown"
        item["states"].add(state)
        item["hashes"].append(digest)
        for c in centres:
            value = (row.get(c) or "").strip()
            if value:
                item["centres"].add(value)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for (gene_id, ident), item in buckets.items():
        clean = item["states"] - {"unknown"}
        state = next(iter(clean)) if len(clean) == 1 and "unknown" not in item["states"] else "unknown"
        grouped[gene_id].append({"id": ident, "state": state, "hashes": sorted(item["hashes"]), "centres": sorted(item["centres"])})

    n = x = u = eligible = 0
    bad_genes: list[str] = []
    unknown_genes: list[str] = []
    gene_centres: dict[str, set[str]] = {}
    for gene_id in sorted(grouped):
        assays = sorted(grouped[gene_id], key=lambda z: (z["id"], z["hashes"]))
        good_idx = next((i for i, z in enumerate(assays) if z["state"] == "good"), None)
        if good_idx is None:
            continue
        rest = [z for i, z in enumerate(assays) if i != good_idx]
        if not rest:
            continue
        eligible += 1
        resolved = [z for z in rest if z["state"] in {"good", "bad"}]
        unknown = [z for z in rest if z["state"] == "unknown"]
        if resolved:
            n += 1
            if any(z["state"] == "bad" for z in resolved):
                x += 1
                bad_genes.append(gene_id)
            gene_centres[gene_id] = {c for z in [assays[good_idx], *resolved] for c in z["centres"]}
        elif unknown:
            u += 1
            unknown_genes.append(gene_id)
    estimate = x / n if n else None
    upper = b_upper(x, n)
    finite = upper - estimate if upper is not None and estimate is not None else None
    residual = u / (n + u) if n + u else None
    total = min(1.0, upper + residual) if upper is not None and residual is not None else None
    bad_set = set(bad_genes)
    all_centres = sorted({c for values in gene_centres.values() for c in values})
    loo = []
    if centres and n:
        for centre in all_centres:
            kept = [g for g, values in gene_centres.items() if centre not in values]
            nn = len(kept)
            xx = sum(g in bad_set for g in kept)
            loo.append({"left_out_centre": centre, "N": nn, "X": xx, "B95": b_upper(xx, nn)})
    loo_available = len(all_centres) >= 2 and len(loo) >= 2
    tests = {"independence_supported": basis is not None, "n_ge_59": n >= 59, "b95_le_0_05": upper is not None and upper <= 0.05, "residual_le_sampling": residual is not None and finite is not None and residual <= finite, "total_upper_le_0_10": total is not None and total <= 0.10, "centre_leave_one_out_available": loo_available, "centre_loo_all_n_ge_59": loo_available and all(z["N"] >= 59 for z in loo), "centre_loo_all_b95_le_0_05": loo_available and all(z["B95"] is not None and z["B95"] <= 0.05 for z in loo)}
    if not loo_available:
        decision, blocker = "BLOCKED", "Mandatory centre leave-one-out kill test cannot be run from source fields."
    elif all(tests.values()):
        decision, blocker = "SURVIVOR", None
    else:
        decision, blocker = ("FAILED" if n else "INCONCLUSIVE"), "One or more frozen fastest-kill or decision criteria failed."
    return result | {"eligible_gene_count": eligible, "N": n, "X": x, "U": u, "p_hat": estimate, "B95": upper, "sampling_term": finite, "residual_R": residual, "two_term_upper_T95": total, "decision_tests": tests, "decision": decision, "blocker": blocker, "adverse_genes": bad_genes, "unresolved_genes": unknown_genes, "leave_one_centre_out": loo, "safety_verdict": "PRECLINICAL_ONLY_NO_HUMAN_OR_TREATMENT_CLAIM"}


def comparable(a: dict, b: dict) -> dict[str, bool]:
    exact = ["decision", "N", "X", "U", "eligible_gene_count", "independence_basis", "raw_row_count", "unique_row_count", "exact_duplicate_row_count"]
    floats = ["p_hat", "B95", "sampling_term", "residual_R", "two_term_upper_T95"]
    checks = {key: a.get(key) == b.get(key) for key in exact}
    for key in floats:
        av, bv = a.get(key), b.get(key)
        checks[key] = (av is None and bv is None) or (av is not None and bv is not None and math.isclose(float(av), float(bv), rel_tol=0, abs_tol=1e-10))
    checks["source_sha256"] = a["source"]["sha256"] == b["source"]["sha256"]
    checks["schema_columns"] = a["schema"]["columns"] == b["schema"]["columns"]
    return checks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        raw, used_url, errors = download()
    except Exception as exc:
        failure = {"execution_state": "FAILED", "stage": "download", "error": repr(exc), "hf_jobs_failure": "402 Payment Required; insufficient prepaid credits", "private_actions_failure": "runner failed before steps; no logs produced"}
        (OUT / "download_failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print("NV01_RESULT_JSON=" + json.dumps(failure, sort_keys=True))
        raise

    a = path_a(raw, used_url)
    b = path_b(raw, used_url)
    checks = comparable(a, b)
    reproduced = all(checks.values())
    reproduction = {"analysis_id": "NV-01", "prereg_sha256": PREREG_SHA256, "execution_state": "REPRODUCED" if reproduced else "FAILED", "decision": a.get("decision") if reproduced else "FAILED_REPRODUCTION_MISMATCH", "checks": checks, "source_download_fallback_errors": errors, "runner": "GitHub Actions public repository", "run_id": os.getenv("GITHUB_RUN_ID"), "commit": os.getenv("GITHUB_SHA")}
    (OUT / "nv01_path_a.json").write_text(json.dumps(a, indent=2, sort_keys=True) + "\n")
    (OUT / "nv01_path_b.json").write_text(json.dumps(b, indent=2, sort_keys=True) + "\n")
    (OUT / "nv01_reproduction.json").write_text(json.dumps(reproduction, indent=2, sort_keys=True) + "\n")
    source = a["source"]
    md = f"""# NV-01 executed result\n\n- Preregistration SHA-256: `{PREREG_SHA256}`\n- Primary state: `EXECUTED`\n- Independent reproduction: `{'REPRODUCED' if reproduced else 'FAILED'}`\n- Decision: `{reproduction['decision']}`\n- Source SHA-256: `{source['sha256']}`\n- Source size: `{source['size_bytes']}` bytes\n- Raw rows: `{a.get('raw_row_count')}`\n- Columns: `{a.get('schema', {}).get('columns')}`\n- Independence basis: `{a.get('independence_basis')}`\n- `N/X/U`: `{a.get('N')}/{a.get('X')}/{a.get('U')}`\n- `B95`: `{a.get('B95')}`\n- Residual `R`: `{a.get('residual_R')}`\n- Total `T95`: `{a.get('two_term_upper_T95')}`\n- Binding blocker: `{a.get('blocker')}`\n- Safety: `PRECLINICAL_ONLY_NO_HUMAN_OR_TREATMENT_CLAIM`\n\n## Barrier certificate\n\n- Objective: certify adverse discordance after an initial viable IMPC call using explicit independent allele or line assessments.\n- Minimum kernel: gene ID, viability class, independent allele/line ID, and phenotyping centre at row level.\n- Binding assumption: repeated rows cannot be treated as independent lines without an explicit identity field.\n- Alternative observable: IMPC experiment-level API or raw exports containing allele accession and centre.\n- Price of possibility: public raw-data reconstruction; no wet lab needed for the first retry.\n- Exact next attack: reconstruct `IMPC_VIA_001` / `IMPC_VIA_001_001` experiment records by gene, allele and centre, then rerun unchanged.\n\nNo human safety, clinical utility, treatment, reproductive, or germline claim follows.\n"""
    (OUT / "NV01_RESULT.md").write_text(md)
    all_hashes = []
    for path in sorted(OUT.glob("*")):
        if path.name != "SHA256SUMS.txt" and path.is_file():
            all_hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(all_hashes) + "\n")
    headline = {"execution_state": reproduction["execution_state"], "decision": reproduction["decision"], "source_sha256": source["sha256"], "source_size_bytes": source["size_bytes"], "raw_row_count": a.get("raw_row_count"), "headers": a.get("schema", {}).get("columns"), "independence_basis": a.get("independence_basis"), "N": a.get("N"), "X": a.get("X"), "U": a.get("U"), "B95": a.get("B95"), "residual_R": a.get("residual_R"), "two_term_upper_T95": a.get("two_term_upper_T95"), "checks": checks}
    print("NV01_RESULT_JSON=" + json.dumps(headline, sort_keys=True))
    if not reproduced:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
