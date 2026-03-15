from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class PerfScenario:
    endpoint: str
    payload: dict[str, Any]
    method: str = "POST"


SCENARIOS: tuple[PerfScenario, ...] = (
    PerfScenario(
        endpoint="/v1/sessions/h2h",
        payload={"num_hands": 100, "stacks": [100, 100], "seed_base": 30_000},
    ),
    PerfScenario(
        endpoint="/v1/benchmark/smoke",
        payload={"num_hands": 50, "stacks": [100, 100], "seed_base": 10_000},
    ),
    PerfScenario(
        endpoint="/v1/benchmark/adaptive",
        payload={"num_hands": 40, "stacks": [100, 100], "seed_base": 50_000},
    ),
    PerfScenario(
        endpoint="/v1/benchmark/h2h",
        payload={
            "num_matches": 3,
            "hands_per_match": 50,
            "stacks": [100, 100],
            "seed_base": 20_000,
        },
    ),
    PerfScenario(
        endpoint="/v1/tournaments/round-robin",
        payload={
            "entrants": ["baseline", "adaptive"],
            "hands_per_match": 20,
            "seed_base": 70_000,
            "stacks": [100, 100],
        },
    ),
)


def _p95(samples: list[float]) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _run_once(client: TestClient, scenario: PerfScenario) -> float:
    start = time.perf_counter()
    if scenario.method == "POST":
        response = client.post(scenario.endpoint, json=scenario.payload)
    else:
        response = client.get(scenario.endpoint)
    elapsed = time.perf_counter() - start
    if response.status_code >= 400:
        raise RuntimeError(
            f"{scenario.endpoint} failed with {response.status_code}: {response.text}"
        )
    return elapsed


def run_perf_gate(
    *,
    warmups: int,
    runs: int,
    p95_limit_seconds: float,
) -> dict[str, Any]:
    from apps.api.main import create_app

    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perf_gate.db"
        app = create_app(db_path=str(db_path))
        client = TestClient(app)
        rows: list[dict[str, Any]] = []
        failed: list[str] = []

        for scenario in SCENARIOS:
            for _ in range(warmups):
                _run_once(client, scenario)
            samples = [_run_once(client, scenario) for _ in range(runs)]
            p95_value = _p95(samples)
            row = {
                "endpoint": scenario.endpoint,
                "runs": runs,
                "warmups": warmups,
                "avg_s": round(statistics.mean(samples), 4),
                "p95_s": round(p95_value, 4),
                "max_s": round(max(samples), 4),
                "limit_s": p95_limit_seconds,
                "pass": p95_value < p95_limit_seconds,
            }
            if not row["pass"]:
                failed.append(scenario.endpoint)
            rows.append(row)

        return {
            "summary": {"pass": not failed, "failed_endpoints": failed},
            "rows": rows,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run /v1 p95 performance gate.")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--p95-limit", type=float, default=3.0)
    parser.add_argument(
        "--output-json",
        default="var/reports/perf_gate_v1.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    report = run_perf_gate(
        warmups=args.warmups,
        runs=args.runs,
        p95_limit_seconds=args.p95_limit,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
