#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import pandas as pd

PREREG_SHA256 = "2b7adb7ee3c464db9188bbf2b4bc133b1338a53d05426c2be64ac4ea5f0fa9bf"
OUT = Path("research/p1_r11_source_gate/results")
OUT.mkdir(parents=True, exist_ok=True)
USER_AGENT = "P1-R11-independent-source-gate/1.0"
MAX_BYTES = 250_000_000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, timeout: int = 120) -> tuple[bytes, dict[str, Any]]:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urlopen(req, timeout=timeout) as response:
        data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise RuntimeError(f"download exceeds {MAX_BYTES} bytes")
        return data, {
            "requested_url": url,
            "final_url": response.geturl(),
            "status": getattr(response, "status", None),
            "headers": dict(response.headers.items()),
            "size_bytes": len(data),
            "sha256": sha256(data),
        }


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value.strip())


def html_links_a(text: str, base: str) -> list[str]:
    parser = LinkParser()
    parser.feed(text)
    return sorted({urljoin(base, x) for x in parser.links})


def html_links_b(text: str, base: str) -> list[str]:
    raw = re.findall(r"(?is)<a\b[^>]*?\bhref\s*=\s*([\"'])(.*?)\1", text)
    return sorted({urljoin(base, value.strip()) for _, value in raw if value.strip()})


def normalize_header(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def choose_columns(columns: list[str], token_groups: list[tuple[str, ...]]) -> list[str]:
    selected: list[str] = []
    for column in columns:
        norm = normalize_header(column)
        if any(all(token in norm for token in group) for group in token_groups):
            selected.append(column)
    return selected


def is_sequence(value: Any) -> bool:
    text = re.sub(r"\s+", "", str(value).upper())
    return 17 <= len(text) <= 35 and bool(re.fullmatch(r"[ACGTUN]+", text))


def canonical_guide(value: Any) -> str | None:
    text = re.sub(r"\s+", "", str(value).upper())
    if not is_sequence(text):
        return None
    return text.replace("U", "T")


def read_table_from_bytes(name: str, data: bytes) -> list[tuple[str, pd.DataFrame]]:
    lower = name.lower()
    tables: list[tuple[str, pd.DataFrame]] = []
    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        payload = gzip.decompress(data) if lower.endswith(".gz") else data
        tables.append((name, pd.read_csv(io.BytesIO(payload), dtype=str, low_memory=False)))
    elif lower.endswith(".tsv") or lower.endswith(".txt") or lower.endswith(".tsv.gz") or lower.endswith(".txt.gz"):
        payload = gzip.decompress(data) if lower.endswith(".gz") else data
        tables.append((name, pd.read_csv(io.BytesIO(payload), sep="\t", dtype=str, low_memory=False)))
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        book = pd.ExcelFile(io.BytesIO(data))
        for sheet in book.sheet_names:
            tables.append((f"{name}::{sheet}", book.parse(sheet, dtype=str)))
    elif lower.endswith(".zip") or data.startswith(b"PK"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in sorted(archive.namelist()):
                if member.endswith("/"):
                    continue
                member_data = archive.read(member)
                try:
                    tables.extend(read_table_from_bytes(member, member_data))
                except Exception:
                    continue
    return tables


def summarize_table(label: str, frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.copy()
    frame.columns = [str(c) for c in frame.columns]
    columns = list(frame.columns)
    guide_cols = choose_columns(columns, [
        ("guide", "sequence"), ("grna",), ("sgrna",), ("guide",),
        ("on target", "sequence"), ("ontarget", "sequence"),
    ])
    study_cols = choose_columns(columns, [
        ("pmid",), ("study",), ("publication",), ("paper",), ("reference",),
    ])
    assay_cols = choose_columns(columns, [
        ("technology",), ("tech",), ("assay",), ("method",),
    ])
    validation_cols = choose_columns(columns, [
        ("validation",), ("validated",), ("cleavage",), ("indel",),
        ("activity",), ("targeted", "result"),
    ])

    guide_quality: dict[str, dict[str, Any]] = {}
    for col in guide_cols:
        values = frame[col].dropna().astype(str)
        seqs = {g for g in (canonical_guide(v) for v in values) if g}
        guide_quality[col] = {
            "nonempty": int(values.size),
            "sequence_like": len(seqs),
            "sequence_fraction": (len(seqs) / max(1, values.nunique())),
        }
    guide_col = None
    if guide_quality:
        guide_col = max(guide_quality, key=lambda c: (guide_quality[c]["sequence_like"], guide_quality[c]["nonempty"]))

    explicit_validation_col = validation_cols[0] if validation_cols else None
    study_col = study_cols[0] if study_cols else None
    assay_col = assay_cols[0] if assay_cols else None

    strata: list[dict[str, Any]] = []
    if guide_col and explicit_validation_col:
        work = frame[[c for c in [guide_col, study_col, assay_col, explicit_validation_col] if c]].copy()
        work["__guide"] = work[guide_col].map(canonical_guide)
        work = work[work["__guide"].notna()]
        work["__validated_nonempty"] = work[explicit_validation_col].fillna("").astype(str).str.strip().ne("")
        group_columns = [c for c in [study_col, assay_col] if c]
        if group_columns:
            for keys, group in work.groupby(group_columns, dropna=False):
                keys_tuple = keys if isinstance(keys, tuple) else (keys,)
                key_map = {group_columns[i]: (None if pd.isna(keys_tuple[i]) else str(keys_tuple[i])) for i in range(len(group_columns))}
                validated = group[group["__validated_nonempty"]]
                strata.append({
                    **key_map,
                    "n_rows": int(len(group)),
                    "n_guides": int(group["__guide"].nunique()),
                    "n_guides_with_explicit_validation": int(validated["__guide"].nunique()),
                    "n_rows_with_explicit_validation": int(len(validated)),
                })
        else:
            validated = work[work["__validated_nonempty"]]
            strata.append({
                "n_rows": int(len(work)),
                "n_guides": int(work["__guide"].nunique()),
                "n_guides_with_explicit_validation": int(validated["__guide"].nunique()),
                "n_rows_with_explicit_validation": int(len(validated)),
            })
    strata = sorted(strata, key=lambda x: (x.get("n_guides_with_explicit_validation", 0), x.get("n_guides", 0)), reverse=True)

    return {
        "label": label,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": columns,
        "guide_columns": guide_cols,
        "study_columns": study_cols,
        "assay_columns": assay_cols,
        "validation_columns": validation_cols,
        "selected_guide_column": guide_col,
        "selected_study_column": study_col,
        "selected_assay_column": assay_col,
        "selected_validation_column": explicit_validation_col,
        "guide_column_diagnostics": guide_quality,
        "largest_strata": strata[:50],
    }


def inspect_gse() -> dict[str, Any]:
    urls = [
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE286300&targ=self&view=brief&form=text",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE286300&targ=all&view=brief&form=text",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE286nnn/GSE286300/soft/GSE286300_family.soft.gz",
    ]
    attempts: list[dict[str, Any]] = []
    texts: list[str] = []
    for url in urls:
        try:
            data, meta = fetch(url)
            entry = {"state": "DATA_ACQUIRED", **meta}
            if data.startswith(b"\x1f\x8b"):
                text = gzip.decompress(data).decode("utf-8", "replace")
            else:
                text = data.decode("utf-8", "replace")
            entry["text_preview"] = text[:2000]
            attempts.append(entry)
            texts.append(text)
        except Exception as exc:
            attempts.append({"state": "FAILED", "url": url, "error": repr(exc)})
    combined = "\n".join(texts)
    lower = combined.lower()
    private_tokens = ["private", "not yet available", "scheduled release", "embargo"]
    is_private = any(token in lower for token in private_tokens)
    sample_accessions = sorted(set(re.findall(r"\bGSM\d+\b", combined)))
    titles = re.findall(r"(?m)^!Series_title\s*=\s*(.+)$", combined)
    release_matches = re.findall(r"(?i)(?:release|available|public)[^\n]{0,80}(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}-\w{3}-\d{4})", combined)
    return {
        "accession": "GSE286300",
        "attempts": attempts,
        "private_or_embargoed_text_detected": is_private,
        "sample_accession_count_in_acquired_metadata": len(sample_accessions),
        "sample_accessions": sample_accessions,
        "series_titles": titles,
        "release_date_strings": release_matches,
        "public_metadata_acquired": bool(texts),
    }


def inspect_crisprofft() -> dict[str, Any]:
    page_url = "https://ccsm.uth.edu/CRISPRoffT/download.html"
    errors: list[dict[str, Any]] = []
    try:
        page_data, page_meta = fetch(page_url)
    except Exception as exc:
        return {"state": "FAILED_ACQUISITION", "page_url": page_url, "errors": [{"url": page_url, "error": repr(exc)}]}
    page_text = page_data.decode("utf-8", "replace")
    links_a = html_links_a(page_text, page_url)
    links_b = html_links_b(page_text, page_url)
    candidate_links = sorted({
        link for link in set(links_a) | set(links_b)
        if urlparse(link).netloc.endswith("uth.edu")
        and (
            re.search(r"\.(csv|tsv|txt|xlsx|xls|zip|gz)(?:$|\?)", link, re.I)
            or "download" in link.lower()
        )
        and link != page_url
    })
    downloads: list[dict[str, Any]] = []
    table_summaries: list[dict[str, Any]] = []
    for link in candidate_links:
        try:
            data, meta = fetch(link)
            name = Path(urlparse(meta["final_url"]).path).name or Path(urlparse(link).path).name or "download"
            record: dict[str, Any] = {"state": "DATA_ACQUIRED", "name": name, **meta}
            try:
                tables = read_table_from_bytes(name, data)
                record["parsed_table_count"] = len(tables)
                for label, frame in tables:
                    try:
                        table_summaries.append(summarize_table(label, frame))
                    except Exception as exc:
                        errors.append({"url": link, "stage": f"summarize:{label}", "error": repr(exc)})
            except Exception as exc:
                record["parse_error"] = repr(exc)
            downloads.append(record)
        except Exception as exc:
            errors.append({"url": link, "stage": "download", "error": repr(exc)})

    qualified: list[dict[str, Any]] = []
    for table in table_summaries:
        for stratum in table.get("largest_strata", []):
            if stratum.get("n_guides_with_explicit_validation", 0) >= 59:
                qualified.append({"table": table["label"], **stratum})
    qualified = sorted(qualified, key=lambda x: x.get("n_guides_with_explicit_validation", 0), reverse=True)
    largest = sorted(
        [
            {"table": table["label"], **stratum}
            for table in table_summaries
            for stratum in table.get("largest_strata", [])
        ],
        key=lambda x: (x.get("n_guides_with_explicit_validation", 0), x.get("n_guides", 0)),
        reverse=True,
    )[:100]

    return {
        "state": "DATA_ACQUIRED",
        "page": page_meta,
        "html_link_parser_agreement": links_a == links_b,
        "links_path_a": links_a,
        "links_path_b": links_b,
        "candidate_download_links": candidate_links,
        "downloads": downloads,
        "table_summaries": table_summaries,
        "largest_study_assay_strata": largest,
        "strata_reaching_59_validated_guides_before_overlap_audit": qualified,
        "errors": errors,
    }


def main() -> None:
    gse = inspect_gse()
    crisprofft = inspect_crisprofft()

    qualified = crisprofft.get("strata_reaching_59_validated_guides_before_overlap_audit", [])
    gse_private = bool(gse.get("private_or_embargoed_text_detected"))
    if qualified:
        decision = "INCONCLUSIVE_OVERLAP_AUDIT_REQUIRED"
        rationale = "At least one apparent 59-guide validated stratum exists, but source-study overlap with P1 must be resolved before calling it independent."
    elif crisprofft.get("table_summaries"):
        decision = "HETEROGENEOUS_RESOURCE_ONLY"
        rationale = "Downloadable tables were acquired, but no parsed coherent study/assay stratum reached 59 biological guides with explicit validation labels."
    elif crisprofft.get("state") == "FAILED_ACQUISITION":
        decision = "FAILED_ACQUISITION"
        rationale = "CRISPRoffT official download page could not be acquired."
    else:
        decision = "BLOCKED_SCHEMA"
        rationale = "Official source was acquired but no parseable table exposed the frozen guide/study/validation kernel."

    if gse_private and decision not in {"EXECUTABLE_REPLICATION_ROUTE", "INCONCLUSIVE_OVERLAP_AUDIT_REQUIRED"}:
        rationale += " GSE286300 was also private or embargoed on the execution date."

    result = {
        "analysis_id": "P1-R11-SOURCE-GATE",
        "prereg_sha256": PREREG_SHA256,
        "execution_state": "EXECUTED",
        "decision": decision,
        "rationale": rationale,
        "gse286300": gse,
        "crisprofft": crisprofft,
        "qualification_floor_distinct_biological_guides": 59,
        "scientific_result_computed": False,
        "safety_verdict": "DATA_READINESS_ONLY_NO_EDITING_OR_TREATMENT_CLAIM",
    }
    out_json = OUT / "P1_R11_SOURCE_GATE_RESULT.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    largest = crisprofft.get("largest_study_assay_strata", [])[:10]
    md = [
        "# P1 hostile panel R11 — independent-source replication gate result",
        "",
        f"- **Preregistration SHA-256:** `{PREREG_SHA256}`.",
        "- **Execution:** `EXECUTED`.",
        f"- **Decision:** `{decision}`.",
        "- **Scientific FNR result:** not computed.",
        "- **Safety:** data-readiness only; no editing or treatment claim.",
        "",
        "## GSE286300",
        "",
        f"- Private/embargo text detected: `{gse_private}`.",
        f"- Public metadata acquired: `{gse.get('public_metadata_acquired')}`.",
        f"- GSM accessions visible in acquired metadata: `{gse.get('sample_accession_count_in_acquired_metadata')}`.",
        f"- Titles: `{gse.get('series_titles')}`.",
        f"- Release-date strings: `{gse.get('release_date_strings')}`.",
        "",
        "## CRISPRoffT source census",
        "",
        f"- Download-page SHA-256: `{crisprofft.get('page', {}).get('sha256')}`.",
        f"- Independent HTML-link parser agreement: `{crisprofft.get('html_link_parser_agreement')}`.",
        f"- Candidate download links: `{len(crisprofft.get('candidate_download_links', []))}`.",
        f"- Acquired downloads: `{len(crisprofft.get('downloads', []))}`.",
        f"- Parsed tables/sheets: `{len(crisprofft.get('table_summaries', []))}`.",
        f"- Apparent strata reaching 59 validated guides before overlap audit: `{len(qualified)}`.",
        "",
        "## Largest parsed study/assay strata",
        "",
        "| Table | Study | Assay | Guides | Guides with explicit validation | Rows |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in largest:
        study = next((row.get(k) for k in row if "pmid" in normalize_header(k) or "study" in normalize_header(k) or "paper" in normalize_header(k) or "publication" in normalize_header(k)), None)
        assay = next((row.get(k) for k in row if "tech" in normalize_header(k) or "assay" in normalize_header(k) or "method" in normalize_header(k)), None)
        md.append(f"| `{row.get('table')}` | `{study}` | `{assay}` | {row.get('n_guides')} | {row.get('n_guides_with_explicit_validation')} | {row.get('n_rows')} |")
    md += [
        "",
        "## Decision",
        "",
        rationale,
        "",
        "A heterogeneous database does not become independent replication merely by aggregating original studies. Any apparent qualifying stratum still requires a source-overlap audit against P1's construction data before execution.",
    ]
    (OUT / "P1_R11_SOURCE_GATE_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    manifest_lines = []
    for path in sorted(OUT.glob("*")):
        if path.name == "SHA256SUMS.txt":
            continue
        manifest_lines.append(f"{sha256(path.read_bytes())}  {path.name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "gse_private": gse_private, "parsed_tables": len(crisprofft.get('table_summaries', [])), "qualified_strata": len(qualified)}, sort_keys=True))


if __name__ == "__main__":
    main()
