from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def full_eda_payload(tmp_path_factory):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "infra" / "scripts" / "mr_poker_full_eda.py"

    output_dir = tmp_path_factory.mktemp("full_eda")
    output_json = output_dir / "mr_poker_full_io_features.json"
    output_md = output_dir / "101_Exploratory_Data_Analysis_Full_IO_Features.md"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--no-include-var-runtime",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"full EDA extractor failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert output_json.exists()
    assert output_md.exists()

    return json.loads(output_json.read_text(encoding="utf-8"))


def test_api_routes_coverage_is_complete(full_eda_payload) -> None:
    coverage = full_eda_payload["coverage"]
    assert coverage["api_routes_total"] == coverage["api_routes_covered"]


def test_mandatory_review_endpoint_is_present(full_eda_payload) -> None:
    api_contracts = full_eda_payload["api_contracts"]
    assert any(
        item["method"] == "GET" and item["path"] == "/v1/hands/{hand_id}/review"
        for item in api_contracts
    )


def test_request_models_coverage_is_complete(full_eda_payload) -> None:
    coverage = full_eda_payload["coverage"]
    request_models = full_eda_payload["api_request_models"]
    assert coverage["request_models_total"] == coverage["request_models_covered"]
    assert len(request_models) == coverage["request_models_total"]
    assert all(model["fields"] for model in request_models)


def test_feature_blocks_coverage_and_required_blocks(full_eda_payload) -> None:
    coverage = full_eda_payload["coverage"]
    feature_coverage = full_eda_payload["feature_catalog"]["coverage"]
    assert coverage["feature_blocks_total"] == coverage["feature_blocks_covered"]
    assert set(feature_coverage["required_blocks"]) == {"minimum", "baseline", "solver-like"}
    assert feature_coverage["blocks_missing"] == []

    critical_gap_categories = {"api", "request_models", "feature_blocks"}
    critical_gaps = [
        gap for gap in full_eda_payload["gaps"] if gap.get("category") in critical_gap_categories
    ]
    assert critical_gaps == []
