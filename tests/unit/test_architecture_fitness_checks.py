from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_architecture_fitness_checks_pass_on_repo(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "infra" / "scripts" / "architecture_fitness_checks.py"
    output_json = tmp_path / "architecture_conformance_report.json"
    output_md = tmp_path / "architecture_conformance_report.md"

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
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"fitness checks failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert output_json.exists()
    assert output_md.exists()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["summary"]["pass"] is True
    assert payload["summary"]["total_violations"] == 0
    assert {rule["rule_id"] for rule in payload["rules"]} == {"R-001", "R-002", "R-003"}
    assert all(rule["status"] == "pass" for rule in payload["rules"])


def test_architecture_fitness_checks_detects_violations(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "services").mkdir(parents=True)
    (repo_root / "packages").mkdir(parents=True)
    (repo_root / "apps").mkdir(parents=True)

    (repo_root / "services" / "bad_service.py").write_text(
        "from fastapi import APIRouter\nfrom apps.api.main import app\n",
        encoding="utf-8",
    )
    (repo_root / "packages" / "bad_pkg.py").write_text(
        "from services.bad_service import app\n",
        encoding="utf-8",
    )

    source_repo_root = Path(__file__).resolve().parents[2]
    script = source_repo_root / "infra" / "scripts" / "architecture_fitness_checks.py"
    output_json = tmp_path / "bad_architecture_report.json"
    output_md = tmp_path / "bad_architecture_report.md"

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
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["summary"]["pass"] is False
    assert payload["summary"]["total_violations"] >= 3
    assert any(rule["status"] == "fail" for rule in payload["rules"])
