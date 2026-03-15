
from services.experiment_runner.service import ExperimentConfig, ExperimentRunner


def test_experiment_runner_generates_report(tmp_path) -> None:
    runner = ExperimentRunner(report_dir=str(tmp_path))
    result = runner.run_h2h(ExperimentConfig(num_matches=2, hands_per_match=10, seed_base=1234))
    assert result['summary']['total_hands'] == 20
    assert 'report_path' in result
