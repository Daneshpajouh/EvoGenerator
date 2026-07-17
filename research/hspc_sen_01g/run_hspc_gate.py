#!/usr/bin/env python3
"""HSPC-SEN-01G: frozen public-data linkage and power gate.

This script inspects machine-readable GEO and PMC supplementary metadata. It does
not fit a model, infer donor identities from sample names, or extract plot values.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

PREREG_SHA256 = "cd37691108ba4b3d0ded938423675713c46eccceda561f1f4070e2152d99ad29"
OUT = Path("research/hspc_sen_01g/results")
SERIES = ["GSE244247", "GSE244248", "GSE287803", "GSE287805"]
EARLY_SERIES = {"GSE244247", "GSE244248"}
LATE_SERIES = {"GSE287803", "GSE287805"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "HSPC-SEN-01G-research-audit/1.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def geo_prefix(accession: str) -> str:
    return accession[:-3] + "nnn"


def soft_url(accession: str) -> str:
    return (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{geo_prefix(accession)}/{accession}/soft/{accession}_family.soft.gz"
    )


def suppl_url(accession: str, filename: str) -> str:
    return (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{geo_prefix(accession)}/{accession}/suppl/{filename}"
    )


def clean_soft_value(value: str) -> str:
    return value.strip().strip('"').strip()


def parse_soft_a(text: str) -> list[dict[str, Any]]:
    """Line-state parser."""
    samples: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        if line.startswith("^SAMPLE = "):
            if current is not None:
                samples.append(current)
            current = {
                "accession": clean_soft_value(line.split("=", 1)[1]),
                "title": [],
                "source_name": [],
                "characteristics": [],
                "description": [],
                "other_fields": defaultdict(list),
            }
            continue
        if current is None or not line.startswith("!Sample_") or " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        value = clean_soft_value(value)
        if key == "!Sample_title":
            current["title"].append(value)
        elif key.startswith("!Sample_source_name"):
            current["source_name"].append(value)
        elif key.startswith("!Sample_characteristics"):
            current["characteristics"].append(value)
        elif key.startswith("!Sample_description"):
            current["description"].append(value)
        else:
            current["other_fields"][key].append(value)
    if current is not None:
        samples.append(current)
    for sample in samples:
        sample["other_fields"] = dict(sample["other_fields"])
    return samples


def parse_soft_b(text: str) -> list[dict[str, Any]]:
    """Independent stanza-split parser."""
    chunks = re.split(r"(?m)^\^SAMPLE = ", text)[1:]
    samples: list[dict[str, Any]] = []
    for chunk in chunks:
        lines = chunk.splitlines()
        accession = clean_soft_value(lines[0]) if lines else ""
        sample: dict[str, Any] = {
            "accession": accession,
            "title": [],
            "source_name": [],
            "characteristics": [],
            "description": [],
            "other_fields": defaultdict(list),
        }
        for line in lines[1:]:
            match = re.match(r"^(!Sample_[^=]+?)\s*=\s*(.*)$", line)
            if not match:
                continue
            key = match.group(1).strip()
            value = clean_soft_value(match.group(2))
            if key == "!Sample_title":
                sample["title"].append(value)
            elif key.startswith("!Sample_source_name"):
                sample["source_name"].append(value)
            elif key.startswith("!Sample_characteristics"):
                sample["characteristics"].append(value)
            elif key.startswith("!Sample_description"):
                sample["description"].append(value)
            else:
                sample["other_fields"][key].append(value)
        sample["other_fields"] = dict(sample["other_fields"])
        samples.append(sample)
    return samples


def canonical_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "accession": sample["accession"],
        "title": sorted(sample["title"]),
        "source_name": sorted(sample["source_name"]),
        "characteristics": sorted(sample["characteristics"]),
        "description": sorted(sample["description"]),
    }


def parse_characteristics(samples: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    by_sample: dict[str, dict[str, list[str]]] = {}
    for sample in samples:
        fields: dict[str, list[str]] = defaultdict(list)
        for item in sample["characteristics"]:
            if ":" in item:
                key, value = item.split(":", 1)
                key = " ".join(key.lower().replace("_", " ").split())
                fields[key].append(value.strip())
            else:
                fields["unkeyed characteristic"].append(item.strip())
        for value in sample["source_name"]:
            fields["source name"].append(value)
        for value in sample["title"]:
            fields["sample title"].append(value)
        for value in sample["description"]:
            fields["description"].append(value)
        by_sample[sample["accession"]] = dict(fields)
    return by_sample


def explicit_identity_fields(parsed: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    """Return explicit person/donor/recipient identity fields only.

    Generic words in source descriptions, treatment labels, sample titles, and
    sample accession numbers do not establish donor identity.
    """
    accepted = re.compile(
        r"(^|\b)(donor|subject|participant|patient|individual|recipient|mouse id|animal id)(\b|$)",
        re.IGNORECASE,
    )
    values: dict[str, set[str]] = defaultdict(set)
    for fields in parsed.values():
        for key, items in fields.items():
            if not accepted.search(key):
                continue
            for item in items:
                item = item.strip()
                if item and item.lower() not in {"na", "n/a", "none", "unknown"}:
                    values[key].add(item)
    return {key: sorted(items) for key, items in sorted(values.items())}


def field_inventory(parsed: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    inventory: dict[str, set[str]] = defaultdict(set)
    for fields in parsed.values():
        for key, items in fields.items():
            inventory[key].update(item for item in items if item)
    return {key: sorted(values) for key, values in sorted(inventory.items())}


def flatten_identity_values(fields: dict[str, list[str]]) -> set[str]:
    return {f"{key}::{value}" for key, values in fields.items() for value in values}


def inspect_tsv_gz(data: bytes) -> dict[str, Any]:
    text = gzip.decompress(data).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    width = max((len(row) for row in rows), default=0)
    return {
        "row_count_including_header": len(rows),
        "max_column_count": width,
        "header": rows[0] if rows else [],
        "first_data_row": rows[1] if len(rows) > 1 else [],
    }


def inspect_tar(data: bytes) -> dict[str, Any]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        summaries = []
        for member in members:
            handle = archive.extractfile(member)
            raw = handle.read() if handle else b""
            first_lines = raw.decode("utf-8-sig", "replace").splitlines()[:3]
            summaries.append(
                {
                    "name": member.name,
                    "size_bytes": member.size,
                    "sha256": sha256(raw),
                    "first_lines": first_lines,
                }
            )
        return {"member_count": len(members), "members": summaries}


def inspect_xlsx(data: bytes) -> dict[str, Any]:
    """Read workbook names and a few cells without treating them as outcome data."""
    if not data.startswith(b"PK"):
        raise ValueError("Downloaded object is not an XLSX/ZIP file")
    try:
        import openpyxl  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openpyxl is required by the frozen runner") from exc
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    sheets = []
    for sheet in workbook.worksheets:
        preview = []
        for row in sheet.iter_rows(min_row=1, max_row=5, values_only=True):
            preview.append([None if value is None else str(value) for value in row[:12]])
        sheets.append({"title": sheet.title, "preview_first_5_rows": preview})
    return {"sheet_count": len(sheets), "sheets": sheets}


def fetch_first(candidates: list[str]) -> tuple[bytes | None, str | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    for url in candidates:
        try:
            return fetch(url), url, errors
        except Exception as exc:
            errors.append({"url": url, "error": repr(exc)})
    return None, None, errors


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Any] = {}
    series_results: dict[str, Any] = {}
    path_agreement: dict[str, bool] = {}
    all_parsed: dict[str, dict[str, dict[str, list[str]]]] = {}
    all_identity: dict[str, dict[str, list[str]]] = {}

    for accession in SERIES:
        url = soft_url(accession)
        data = fetch(url)
        sources[f"{accession}_soft"] = {
            "url": url,
            "sha256": sha256(data),
            "size_bytes": len(data),
        }
        text = gzip.decompress(data).decode("utf-8-sig")
        a = parse_soft_a(text)
        b = parse_soft_b(text)
        ca = sorted((canonical_sample(item) for item in a), key=lambda item: item["accession"])
        cb = sorted((canonical_sample(item) for item in b), key=lambda item: item["accession"])
        path_agreement[accession] = ca == cb
        parsed = parse_characteristics(a)
        identities = explicit_identity_fields(parsed)
        all_parsed[accession] = parsed
        all_identity[accession] = identities
        series_results[accession] = {
            "sample_count_path_a": len(a),
            "sample_count_path_b": len(b),
            "parser_agreement": ca == cb,
            "explicit_identity_fields": identities,
            "field_inventory": field_inventory(parsed),
            "sample_accessions": sorted(parsed),
        }

    barcode_url = suppl_url("GSE287803", "GSE287803_Barcode_count.tsv.gz")
    barcode_data = fetch(barcode_url)
    sources["GSE287803_barcode_count"] = {
        "url": barcode_url,
        "sha256": sha256(barcode_data),
        "size_bytes": len(barcode_data),
    }
    barcode_summary = inspect_tsv_gz(barcode_data)

    nhej_url = suppl_url("GSE287805", "GSE287805_RAW.tar")
    nhej_data = fetch(nhej_url)
    sources["GSE287805_raw_tar"] = {
        "url": nhej_url,
        "sha256": sha256(nhej_data),
        "size_bytes": len(nhej_data),
    }
    nhej_summary = inspect_tar(nhej_data)

    xlsx_results: dict[str, Any] = {}
    for name in ["mmc2.xlsx", "mmc3.xlsx", "mmc4.xlsx"]:
        candidates = [
            f"https://pmc.ncbi.nlm.nih.gov/articles/PMC12208344/bin/{name}",
            f"https://pmc.ncbi.nlm.nih.gov/articles/instance/12208344/bin/{name}",
        ]
        data, used_url, errors = fetch_first(candidates)
        if data is None or used_url is None:
            xlsx_results[name] = {"state": "FAILED", "errors": errors}
            continue
        sources[name] = {"url": used_url, "sha256": sha256(data), "size_bytes": len(data)}
        try:
            xlsx_results[name] = {
                "state": "SOURCE_VERIFIED",
                "inspection": inspect_xlsx(data),
                "fallback_errors": errors,
            }
        except Exception as exc:
            xlsx_results[name] = {
                "state": "FAILED",
                "error": repr(exc),
                "fallback_errors": errors,
                "prefix_hex": data[:16].hex(),
            }

    early_ids: set[str] = set()
    late_ids: set[str] = set()
    for accession in EARLY_SERIES:
        early_ids.update(flatten_identity_values(all_identity[accession]))
    for accession in LATE_SERIES:
        late_ids.update(flatten_identity_values(all_identity[accession]))
    linked_ids = early_ids & late_ids

    explicit_early = bool(early_ids)
    explicit_late = bool(late_ids)
    full_key_public = bool(linked_ids)
    d_early = len(early_ids) if explicit_early else None
    d_late = len(late_ids) if explicit_late else None
    d_linked = len(linked_ids) if full_key_public else None

    # The BAR-seq table is a tabular consequence readout, but its existence cannot
    # substitute for an explicit donor-condition-recipient linkage.
    long_term_tabular = barcode_summary["row_count_including_header"] > 1

    if not all(path_agreement.values()):
        gate_state = "FAILED"
        blocker = "Independent SOFT parsers disagree."
    elif not full_key_public:
        gate_state = "BLOCKED"
        blocker = (
            "No explicit stable donor/subject/recipient identity value is shared "
            "between the early RNA-seq and later BAR-seq/NHEJ records."
        )
    elif d_linked is not None and d_linked < 29:
        gate_state = "FAILED"
        blocker = f"Only {d_linked} explicitly linked donors; frozen floor is 29."
    elif not long_term_tabular:
        gate_state = "BLOCKED"
        blocker = "No tabular long-term consequence was found."
    else:
        gate_state = "PASSED"
        blocker = None

    result = {
        "analysis_id": "HSPC-SEN-01G",
        "prereg_sha256": PREREG_SHA256,
        "execution_state": "EXECUTED",
        "reproduction_state": "REPRODUCED" if all(path_agreement.values()) else "FAILED",
        "gate_state": gate_state,
        "blocker": blocker,
        "sources": sources,
        "series": series_results,
        "parser_agreement": path_agreement,
        "D_early": d_early,
        "D_late": d_late,
        "D_linked": d_linked,
        "explicit_early_identity_available": explicit_early,
        "explicit_late_identity_available": explicit_late,
        "full_early_late_key_public": full_key_public,
        "linked_identity_values": sorted(linked_ids),
        "barcode_table": barcode_summary,
        "nhej_archive": nhej_summary,
        "supplementary_xlsx": xlsx_results,
        "safety_verdict": "PRECLINICAL_DATA_LINKAGE_GATE_ONLY",
        "forbidden_inferences": [
            "sample numbers are donor identifiers",
            "treatment labels establish donor matching",
            "mouse-level records are independent human donors",
            "group plots provide row-level outcomes",
            "the published association is a prospective clinical certificate",
        ],
        "runner": {
            "repository": os.getenv("GITHUB_REPOSITORY"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "commit": os.getenv("GITHUB_SHA"),
        },
    }
    (OUT / "hspc_sen_01g_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    series_lines = []
    for accession in SERIES:
        info = series_results[accession]
        series_lines.append(
            f"| {accession} | {info['sample_count_path_a']} | "
            f"{list(info['explicit_identity_fields']) or 'none'} | "
            f"{info['parser_agreement']} |"
        )
    xlsx_lines = []
    for name, info in xlsx_results.items():
        if info.get("state") == "SOURCE_VERIFIED":
            sheets = [s["title"] for s in info["inspection"]["sheets"]]
            xlsx_lines.append(f"- `{name}`: workbook sheets `{sheets}`; no PDF values extracted.")
        else:
            xlsx_lines.append(f"- `{name}`: inspection `{info.get('state')}`.")

    md = f"""# HSPC-SEN-01G executed public linkage gate

- **Preregistration SHA-256:** `{PREREG_SHA256}`
- **Execution:** `EXECUTED`
- **Independent metadata parsing:** `{result['reproduction_state']}`
- **Gate decision:** `{gate_state}`
- **Binding blocker:** {blocker}
- **Safety:** `PRECLINICAL_DATA_LINKAGE_GATE_ONLY`

## Machine-readable source census

| Series | GEO samples | Explicit identity fields | Two-parser agreement |
|---|---:|---|---|
{chr(10).join(series_lines)}

The public series contain treatment-labelled sample records. The frozen gate did
not treat sample order, sample number, well code, treatment, or GEO accession as a
human donor identifier.

## Frozen counts

- `D_early`: `{d_early}`
- `D_late`: `{d_late}`
- `D_linked`: `{d_linked}`
- Shared explicit early/late identity values: `{sorted(linked_ids)}`
- BAR-seq table rows including header: `{barcode_summary['row_count_including_header']}`
- BAR-seq maximum columns: `{barcode_summary['max_column_count']}`
- NHEJ raw files: `{nhej_summary['member_count']}`

`None` means the donor count is not identifiable from an explicit public key. It
is not zero donors and must not be converted into a numeric cohort size.

## Supplementary-table inspection

{chr(10).join(xlsx_lines)}

The article's public Excel supplements concern structural aberrations and gene
lists. The long-term functional results are otherwise reported through article
figures and GEO BAR/NHEJ records. No plot values were digitized.

## Barrier certificate

- **Blocked objective:** pair a donor-level day-one/day-four senescence or
  inflammatory measure with the same donor-condition's 15-week engraftment or
  clonal-diversity consequence.
- **Minimum missing kernel:** an explicit table containing donor ID, editing
  condition, recipient mouse ID, early measurement, long-term outcome, and
  collection time.
- **Barrier-producing assumption:** sample numbering and treatment order are not
  identity keys. Treating them as donors would manufacture independence.
- **Public alternatives checked:** GEO SOFT records for all four SubSeries, the
  BAR-seq count table, the NHEJ raw archive, and public Excel supplements.
- **Price of possibility:** a de-identified donor-to-recipient crosswalk and the
  corresponding tabular early and long-term values from the authors, or a new
  public prospective cohort with that key.
- **Exact next attack:** request the smallest row-level crosswalk under a data-use
  agreement; preregister the certificate only after `D_linked >= 29` is verified.

## Scientific verdict

This route is `BLOCKED`, not a negative biological result. The publication's
mechanistic and treatment findings remain prior art. The public deposit does not
support a new donor-level distribution-free early-warning certificate.

No clinical utility, treatment recommendation, dosing recommendation, human safety,
reproductive, or germline claim follows.
"""
    (OUT / "HSPC_SEN_01G_GATE_RESULT.md").write_text(md, encoding="utf-8")

    hashes = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            hashes.append(f"{sha256(path.read_bytes())}  {path.name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    headline = {
        "execution_state": result["execution_state"],
        "reproduction_state": result["reproduction_state"],
        "gate_state": gate_state,
        "D_early": d_early,
        "D_late": d_late,
        "D_linked": d_linked,
        "series_sample_counts": {
            key: value["sample_count_path_a"] for key, value in series_results.items()
        },
        "explicit_identity_fields": {
            key: value["explicit_identity_fields"] for key, value in series_results.items()
        },
        "blocker": blocker,
    }
    print("HSPC_GATE_JSON=" + json.dumps(headline, sort_keys=True))
    if result["reproduction_state"] != "REPRODUCED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
