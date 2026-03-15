
import subprocess
import sys
from pathlib import Path

from services.governance_service import GovernanceService
from services.regression_suite_service import RegressionSuiteService
from services.deploy_service import DeployService
from services.release_notes_service import ReleaseNotesService


def test_deploy_manifest_has_assets():
    manifest = DeployService(base_dir='.')
    payload = manifest.manifest()
    assert payload['dockerfile_present'] is True
    assert payload['compose_present'] is True


def test_regression_suite_build(tmp_path):
    svc = RegressionSuiteService(output_dir=str(tmp_path / 'regression'))
    result = svc.build('suite_x')
    assert result['suite_name'] == 'suite_x'
    assert result['spot_pack_count'] >= 1


def test_release_notes_current():
    svc = ReleaseNotesService(docs_dir='docs')
    notes = svc.current_notes()
    assert notes['release_train'] == 'Sprint 21-30'
    assert len(notes['highlights']) >= 3


def test_audit_consistency_guardrail_script():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / 'infra' / 'scripts' / 'audit_consistency_check.py'
    result = subprocess.run(
        [sys.executable, str(script), '--repo-root', str(repo_root), '--strict'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f'audit consistency check failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}'
