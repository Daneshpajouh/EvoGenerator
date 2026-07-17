#!/usr/bin/env python3
from __future__ import annotations

import html as html_module
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import run_gate as base

PREREG_SHA256 = base.PREREG_SHA256
TECHNICAL_AMENDMENT_A_SHA256 = "d375b511e69fa5d464312d6b773120cf79c10c9ef9d2b82a518a684ce920df49"
DYNAMIC_DOWNLOAD_URL = "https://ccsm.uth.edu/CRISPRoffT/gene_search_result_0.cgi?type=all_dataset_info_download"
OUT = base.OUT


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


def normalize_link(value: str, base_url: str) -> str:
    decoded = html_module.unescape(value.strip())
    decoded = re.sub(r"(?:&amp;?|&)$", "", decoded)
    absolute = urljoin(base_url, decoded)
    parsed = urlparse(absolute)
    return parsed._replace(fragment="").geturl()


def links_path_a(text: str, base_url: str) -> list[str]:
    parser = LinkParser()
    parser.feed(text)
    return sorted({normalize_link(value, base_url) for value in parser.links if value.strip()})


def links_path_b(text: str, base_url: str) -> list[str]:
    raw = re.findall(r"(?is)<a\b[^>]*?\bhref\s*=\s*([\"'])(.*?)\1", text)
    return sorted({normalize_link(value, base_url) for _, value in raw if value.strip()})


def candidate_data_links(links: list[str]) -> list[str]:
    static = {
        link
        for link in links
        if urlparse(link).netloc.endswith("uth.edu")
        and re.search(r"\.(csv|tsv|txt|xlsx|xls|zip|gz)(?:$|\?)", link, re.I)
    }
    static.add(DYNAMIC_DOWNLOAD_URL)
    return sorted(static)


def inspect_crisprofft() -> dict[str, Any]:
    page_url = "https://ccsm.uth.edu/CRISPRoffT/download.html"
    errors: list[dict[str, Any]] = []
    try:
        page_data, page_meta = base.fetch(page_url)
    except Exception as exc:
        return {
            "state": "FAILED_ACQUISITION",
            "page_url": page_url,
            "technical_amendment_a_sha256": TECHNICAL_AMENDMENT_A_SHA256,
            "errors": [{"url": page_url, "error": repr(exc)}],
        }

    page_text = page_data.decode("utf-8", "replace")
    (OUT / "CRISPRoffT_download_page.html").write_bytes(page_data)

    raw_a = links_path_a(page_text, page_url)
    raw_b = links_path_b(page_text, page_url)
    candidate_a = candidate_data_links(raw_a)
    candidate_b = candidate_data_links(raw_b)
    agreement = candidate_a == candidate_b
    candidates = sorted(set(candidate_a) | set(candidate_b))

    downloads: list[dict[str, Any]] = []
    table_summaries: list[dict[str, Any]] = []
    for link in candidates:
        try:
            data, meta = base.fetch(link)
            content_type = str(meta.get("headers", {}).get("Content-Type", "")).lower()
            name = Path(urlparse(meta["final_url"]).path).name or Path(urlparse(link).path).name or "download"
            record: dict[str, Any] = {
                "state": "DATA_ACQUIRED",
                "name": name,
                "content_type": content_type,
                **meta,
            }
            if "text/html" in content_type:
                record["data_download_state"] = "HTML_NOT_TABLE"
                record["parsed_table_count"] = 0
                record["html_preview"] = data.decode("utf-8", "replace")[:1000]
            else:
                try:
                    tables = base.read_table_from_bytes(name, data)
                    record["parsed_table_count"] = len(tables)
                    record["data_download_state"] = "PARSED" if tables else "NO_SUPPORTED_TABLE"
                    for label, frame in tables:
                        try:
                            table_summaries.append(base.summarize_table(label, frame))
                        except Exception as exc:
                            errors.append(
                                {"url": link, "stage": f"summarize:{label}", "error": repr(exc)}
                            )
                except Exception as exc:
                    record["parse_error"] = repr(exc)
                    record["data_download_state"] = "PARSE_FAILED"
            downloads.append(record)
        except Exception as exc:
            errors.append({"url": link, "stage": "download", "error": repr(exc)})

    qualified: list[dict[str, Any]] = []
    for table in table_summaries:
        for stratum in table.get("largest_strata", []):
            if stratum.get("n_guides_with_explicit_validation", 0) >= 59:
                qualified.append({"table": table["label"], **stratum})
    qualified.sort(
        key=lambda row: row.get("n_guides_with_explicit_validation", 0),
        reverse=True,
    )

    largest = sorted(
        [
            {"table": table["label"], **stratum}
            for table in table_summaries
            for stratum in table.get("largest_strata", [])
        ],
        key=lambda row: (
            row.get("n_guides_with_explicit_validation", 0),
            row.get("n_guides", 0),
        ),
        reverse=True,
    )[:100]

    return {
        "state": "DATA_ACQUIRED",
        "page": page_meta,
        "technical_amendment_a_sha256": TECHNICAL_AMENDMENT_A_SHA256,
        "raw_anchor_links_path_a": raw_a,
        "raw_anchor_links_path_b": raw_b,
        "candidate_links_path_a": candidate_a,
        "candidate_links_path_b": candidate_b,
        "candidate_link_parser_agreement": agreement,
        "candidate_download_links": candidates,
        "dynamic_download_url": DYNAMIC_DOWNLOAD_URL,
        "downloads": downloads,
        "table_summaries": table_summaries,
        "largest_study_assay_strata": largest,
        "strata_reaching_59_validated_guides_before_overlap_audit": qualified,
        "errors": errors,
    }


def main() -> None:
    gse = base.inspect_gse()
    crisprofft = inspect_crisprofft()
    qualified = crisprofft.get("strata_reaching_59_validated_guides_before_overlap_audit", [])
    gse_private = bool(gse.get("private_or_embargoed_text_detected"))
    parser_agreement = bool(crisprofft.get("candidate_link_parser_agreement"))

    if crisprofft.get("state") != "FAILED_ACQUISITION" and not parser_agreement:
        decision = "FAILED_REPRODUCTION_MISMATCH"
        rationale = (
            "Independent normalized candidate-link parsers disagreed; "
            "the source gate cannot pass."
        )
    elif qualified:
        decision = "INCONCLUSIVE_OVERLAP_AUDIT_REQUIRED"
        rationale = (
            "At least one apparent 59-guide validated stratum exists, but "
            "source-study overlap with P1 must be resolved before calling it independent."
        )
    elif crisprofft.get("table_summaries"):
        decision = "HETEROGENEOUS_RESOURCE_ONLY"
        rationale = (
            "Downloadable tables were acquired, but no parsed coherent study/assay "
            "stratum reached 59 biological guides with explicit validation labels."
        )
    elif crisprofft.get("state") == "FAILED_ACQUISITION":
        decision = "FAILED_ACQUISITION"
        rationale = "CRISPRoffT official download page could not be acquired."
    else:
        decision = "BLOCKED_SCHEMA"
        rationale = (
            "Official source was acquired but no parseable table exposed the frozen "
            "guide/study/validation kernel."
        )

    if gse_private and decision not in {
        "EXECUTABLE_REPLICATION_ROUTE",
        "INCONCLUSIVE_OVERLAP_AUDIT_REQUIRED",
    }:
        rationale += " GSE286300 was also private or embargoed on the execution date."

    result = {
        "analysis_id": "P1-R11-SOURCE-GATE",
        "prereg_sha256": PREREG_SHA256,
        "technical_amendment_a_sha256": TECHNICAL_AMENDMENT_A_SHA256,
        "execution_state": "REPRODUCED_AFTER_TRANSPORT_REPAIR",
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
        f"- **Technical amendment A SHA-256:** `{TECHNICAL_AMENDMENT_A_SHA256}`.",
        "- **Execution:** `REPRODUCED_AFTER_TRANSPORT_REPAIR`.",
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
        f"- Normalized candidate-link parser agreement: `{parser_agreement}`.",
        f"- Dynamic-download URL: `{DYNAMIC_DOWNLOAD_URL}`.",
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
        study = next(
            (
                row.get(key)
                for key in row
                if "pmid" in base.normalize_header(key)
                or "study" in base.normalize_header(key)
                or "paper" in base.normalize_header(key)
                or "publication" in base.normalize_header(key)
            ),
            None,
        )
        assay = next(
            (
                row.get(key)
                for key in row
                if "tech" in base.normalize_header(key)
                or "assay" in base.normalize_header(key)
                or "method" in base.normalize_header(key)
            ),
            None,
        )
        md.append(
            f"| `{row.get('table')}` | `{study}` | `{assay}` | "
            f"{row.get('n_guides')} | "
            f"{row.get('n_guides_with_explicit_validation')} | "
            f"{row.get('n_rows')} |"
        )
    md += [
        "",
        "## Decision",
        "",
        rationale,
        "",
        "A heterogeneous database does not become independent replication merely by "
        "aggregating original studies. Any apparent qualifying stratum still requires "
        "a source-overlap audit against P1's construction data before execution.",
    ]
    (OUT / "P1_R11_SOURCE_GATE_RESULT.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )

    manifest_lines = []
    for path in sorted(OUT.glob("*")):
        if path.name == "SHA256SUMS.txt":
            continue
        manifest_lines.append(f"{base.sha256(path.read_bytes())}  {path.name}")
    (OUT / "SHA256SUMS.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "gse_private": gse_private,
                "parser_agreement": parser_agreement,
                "parsed_tables": len(crisprofft.get("table_summaries", [])),
                "qualified_strata": len(qualified),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
