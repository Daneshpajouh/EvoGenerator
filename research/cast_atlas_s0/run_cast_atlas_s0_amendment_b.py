#!/usr/bin/env python3
from __future__ import annotations

import io
import itertools
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import openpyxl

import run_cast_atlas_s0 as base
import run_cast_atlas_s0_amendment_a as amend_a

TECHNICAL_AMENDMENT_B_SHA256 = "3fbc6e3bcbb4a91a09ed849aef0be4e80d51c33f7a30b09e34c7664a3b5785b8"


def openpyxl_tables(name: str, raw: bytes) -> list[dict[str, Any]]:
    workbook = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    tables: list[dict[str, Any]] = []
    for sheet in workbook.worksheets:
        if hasattr(sheet, "reset_dimensions"):
            sheet.reset_dimensions()
        rows = [list(row) for row in itertools.islice(sheet.iter_rows(values_only=True), 30)]
        index, headers = base.choose_header(rows)
        nonempty_headers = [header for header in headers if header]
        max_width = max((len(row) for row in rows), default=0)
        tables.append(
            {
                "path": "A_openpyxl_read_only_reset_dimensions",
                "file": name,
                "sheet": sheet.title,
                "header_row_index_zero_based": index,
                "headers": nonempty_headers,
                "field_map": base.classify_headers(nonempty_headers),
                "preview_row_count": len(rows),
                "preview_max_width": max_width,
            }
        )
    workbook.close()
    return tables


def missing_drawing_targets(raw: bytes) -> list[str]:
    warnings: set[str] = set()
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = set(archive.namelist())
            for rel_name in sorted(name for name in names if name.endswith(".rels")):
                try:
                    root = ET.fromstring(archive.read(rel_name))
                except Exception:
                    continue
                for rel in root:
                    target = rel.attrib.get("Target", "")
                    if "drawings/" not in target:
                        continue
                    if target.startswith("/"):
                        resolved = target.lstrip("/")
                    elif rel_name.startswith("xl/worksheets/_rels/"):
                        resolved = "xl/" + target.replace("../", "", 1)
                    elif rel_name.startswith("xl/_rels/"):
                        resolved = "xl/" + target
                    else:
                        resolved = target
                    if resolved not in names:
                        warnings.add(resolved)
    except Exception:
        return []
    return sorted(warnings)


def inspect_archive(accession: str, data: bytes) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    tables_a: list[dict[str, Any]] = []
    tables_b: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    source_warnings: list[dict[str, Any]] = []

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        file_members = sorted(
            (member for member in archive.getmembers() if member.isfile()),
            key=lambda member: member.name,
        )
        for member in file_members:
            handle = archive.extractfile(member)
            raw = handle.read() if handle else b""
            members.append(
                {
                    "name": member.name,
                    "size_bytes": member.size,
                    "sha256": base.sha256(raw),
                }
            )
            lower = member.name.lower()
            if lower.endswith(".xlsx"):
                missing = missing_drawing_targets(raw)
                if missing:
                    source_warnings.append(
                        {
                            "member": member.name,
                            "warning": "MISSING_DRAWING_RELATIONSHIP_TARGET",
                            "targets": missing,
                        }
                    )
                try:
                    tables_a.extend(openpyxl_tables(member.name, raw))
                except Exception as exc:
                    errors.append(
                        {
                            "member": member.name,
                            "parser": "A_openpyxl_read_only_reset_dimensions",
                            "error": repr(exc),
                        }
                    )
                try:
                    tables_b.extend(base.raw_ooxml_tables(member.name, raw))
                except Exception as exc:
                    errors.append(
                        {
                            "member": member.name,
                            "parser": "B_raw_ooxml",
                            "error": repr(exc),
                        }
                    )
            elif lower.endswith((".txt", ".tsv", ".csv")):
                try:
                    parsed = base.text_table(member.name, raw)
                    tables_a.extend([{**row, "path": "A_text_csv"} for row in parsed])
                except Exception as exc:
                    errors.append(
                        {
                            "member": member.name,
                            "parser": "A_text_csv",
                            "error": repr(exc),
                        }
                    )
                try:
                    parsed = base.text_table(member.name, raw)
                    tables_b.extend([{**row, "path": "B_text_csv"} for row in parsed])
                except Exception as exc:
                    errors.append(
                        {
                            "member": member.name,
                            "parser": "B_text_csv",
                            "error": repr(exc),
                        }
                    )

    return {
        "accession": accession,
        "members": members,
        "tables_path_a": tables_a,
        "tables_path_b": tables_b,
        "errors": errors,
        "source_integrity_warnings": source_warnings,
    }


def write_final_outputs() -> None:
    a_path = base.OUT / "CAST_ATLAS_S0_AMENDMENT_A_RESULT.json"
    result = json.loads(a_path.read_text(encoding="utf-8"))
    result["technical_amendment_b_sha256"] = TECHNICAL_AMENDMENT_B_SHA256
    result["execution_state"] = "REPRODUCED_AFTER_TECHNICAL_AMENDMENTS_A_B"

    schema_agreement = all(
        item.get("schema_parser_agreement") is True
        for item in result.get("archive_results", {}).values()
        if item.get("state") != "FAILED"
    )
    path_specific_errors = [
        {"study": study, **error}
        for study, item in result.get("archive_results", {}).items()
        for error in item.get("errors", [])
    ]
    warnings = [
        {"study": study, **warning}
        for study, item in result.get("archive_results", {}).items()
        for warning in item.get("source_integrity_warnings", [])
    ]
    result["workbook_schema_agreement"] = schema_agreement
    result["path_specific_parser_errors"] = path_specific_errors
    result["source_integrity_warnings"] = warnings

    if not schema_agreement or path_specific_errors:
        result["decision"] = "FAILED_REPRODUCTION"
        result["rationale"] = (
            "Workbook paths did not both complete with exact schema agreement."
        )
    elif result.get("treated_condition_count") is not None and result["treated_condition_count"] < 59:
        result["decision"] = "SAMPLE_SIZE_BARRIER"
        result["rationale"] = (
            f"Only {result['treated_condition_count']} independent treated edit conditions "
            "remained after conservative collapse; 59 were required."
        )
    elif len(result.get("compatible_studies", [])) < 2:
        any_tables = any(
            item.get("tables_path_a")
            for item in result.get("archive_results", {}).values()
            if item.get("state") != "FAILED"
        )
        result["decision"] = "SCHEMA_BARRIER" if any_tables else "PROCESSED_TABLE_ABSENT"
        result["rationale"] = (
            "Fewer than two independent studies exposed a compatible event-level table."
        )
    elif not result.get("complete_common_event_key"):
        result["decision"] = "SCHEMA_BARRIER"
        result["rationale"] = (
            "Candidate event tables did not share the complete frozen event key."
        )
    elif not result.get("denominator_measurable"):
        result["decision"] = "DENOMINATOR_BARRIER"
        result["rationale"] = (
            "No predictor-independent missed-event denominator was measurable."
        )
    else:
        result["decision"] = "EXECUTABLE_ATLAS_ROUTE"
        result["rationale"] = (
            "Every frozen schema, sample-size, denominator, and reproduction gate passed."
        )

    final_json = base.OUT / "CAST_ATLAS_S0_AMENDMENT_B_RESULT.json"
    final_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# CAST-ATLAS-S0 — final schema and power gate result",
        "",
        f"- **Public binding SHA-256:** `{base.BINDING_SHA256}`.",
        f"- **Technical amendment A SHA-256:** `{amend_a.TECHNICAL_AMENDMENT_A_SHA256}`.",
        f"- **Technical amendment B SHA-256:** `{TECHNICAL_AMENDMENT_B_SHA256}`.",
        f"- **Execution:** `{result.get('execution_state')}`.",
        f"- **Decision:** `{result.get('decision')}`.",
        "- **Safety:** data-readiness only; no editor-safety or treatment claim.",
        "",
        "## Reproduction",
        "",
        f"- Per-sample condition/exclusion agreement: `{result.get('sample_diagnostic_agreement')}`.",
        f"- Condition-set agreement: `{result.get('condition_parser_agreement')}`.",
        f"- Workbook schema agreement: `{schema_agreement}`.",
        f"- Path-specific parser errors: `{len(path_specific_errors)}`.",
        f"- Source-integrity warnings: `{len(warnings)}`.",
        "",
        "## Power and source census",
        "",
        f"- Independent treated edit conditions: `{result.get('treated_condition_count')}`.",
        f"- Required floor: `59`.",
        f"- Excluded incomplete treated samples: `{result.get('excluded_incomplete_treated_sample_count')}`.",
        f"- Controls excluded: `{result.get('control_sample_count')}`.",
        "",
        "## Event-table gate",
        "",
        "| Study | Archive members | Parsed sheets A | Parsed sheets B | Candidate event tables |",
        "|---|---:|---:|---:|---:|",
    ]
    for study, item in result.get("archive_results", {}).items():
        md.append(
            f"| `{study}` | {len(item.get('members', []))} | "
            f"{len(item.get('tables_path_a', []))} | "
            f"{len(item.get('tables_path_b', []))} | "
            f"{len(result.get('candidate_tables_by_study', {}).get(study, []))} |"
        )
    md += [
        "",
        f"- Compatible studies: `{result.get('compatible_studies')}`.",
        f"- Common classified fields: `{result.get('common_fields')}`.",
        f"- Complete common event key: `{result.get('complete_common_event_key')}`.",
        f"- Predictor-independent missed-event denominator measurable: `{result.get('denominator_measurable')}`.",
        "",
        "## ENA route",
        "",
        f"- PRJEB106504 read-run rows: `{result.get('ena_prjeb106504', {}).get('row_count')}`.",
        f"- Processed event table present: `{result.get('ena_prjeb106504', {}).get('processed_event_table_present')}`.",
        "",
        "## Binding decision",
        "",
        str(result.get("rationale")),
        "",
        "The public studies remain useful for structural-event reconstruction and "
        "positive-event harmonization. They do not yet support the preregistered "
        "leave-study-out certificate unless every frozen gate passes.",
    ]
    (base.OUT / "CAST_ATLAS_S0_AMENDMENT_B_RESULT.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )

    manifest = []
    for path in sorted(base.OUT.glob("*")):
        if path.name != "SHA256SUMS.txt":
            manifest.append(f"{base.sha256(path.read_bytes())}  {path.name}")
    (base.OUT / "SHA256SUMS.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "decision": result.get("decision"),
                "treated_condition_count": result.get("treated_condition_count"),
                "workbook_schema_agreement": schema_agreement,
                "compatible_studies": result.get("compatible_studies"),
                "complete_common_event_key": result.get("complete_common_event_key"),
                "denominator_measurable": result.get("denominator_measurable"),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    base.condition_key_a = amend_a.condition_key_a
    base.condition_key_b = amend_a.condition_key_b
    base.openpyxl_tables = openpyxl_tables
    base.inspect_archive = inspect_archive
    base.main()
    amend_a.write_amended_outputs()
    write_final_outputs()


if __name__ == "__main__":
    main()
