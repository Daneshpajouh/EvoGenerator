#!/usr/bin/env python3
"""Technical transport fix for HSPC-SEN-01G amendment A.

NCBI moved legacy PMC OA packages beneath `/pub/pmc/deprecated/` on
2026-04-13. This wrapper changes only that download path and then executes the
hash-frozen audit unchanged.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import run_pmc_amendment_a as audit

_original_oa_package_url = audit.oa_package_url


def migrated_oa_package_url() -> tuple[str, bytes]:
    url, api_xml = _original_oa_package_url()
    marker = "/pub/pmc/oa_package/"
    if marker in url:
        url = url.replace(marker, "/pub/pmc/deprecated/oa_package/", 1)
    return url, api_xml


def main() -> None:
    audit.oa_package_url = migrated_oa_package_url
    try:
        audit.main()
    except BaseException as exc:
        out = Path("research/hspc_sen_01g/pmc_audit_results")
        out.mkdir(parents=True, exist_ok=True)
        failure = {
            "analysis_id": "HSPC-SEN-01G-AMENDMENT-A",
            "execution_state": "FAILED",
            "amendment_sha256": audit.AMENDMENT_SHA256,
            "parent_prereg_sha256": audit.PARENT_PREREG_SHA256,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        (out / "pmc_amendment_a_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("HSPC_PMC_AUDIT_FAILURE=" + json.dumps(failure, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
