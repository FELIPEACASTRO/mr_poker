from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

SUPPORTED_CAPABILITY_CLASSES = {"mock", "heuristic", "real"}


def _normalize_capability_class(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in SUPPORTED_CAPABILITY_CLASSES:
            return lowered
    return "mock"


class ExternalSolverAdapter:
    def __init__(self, base_dir: str = "var/external_solver") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def catalog(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.catalog_dto()]

    def catalog_dto(self) -> list["ExternalSolverCatalogEntry"]:
        result: list[ExternalSolverCatalogEntry] = []
        for path in sorted(self.base_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            capability_class = _normalize_capability_class(
                payload.get("capability_class")
            )
            result.append(
                ExternalSolverCatalogEntry(
                    name=str(payload.get("name", path.stem)),
                    path=str(path),
                    label_count=len(payload.get("labels", [])),
                    capability_class=capability_class,
                )
            )
        return result

    def load(self, name: str) -> dict[str, Any]:
        path = self.base_dir / f"{name}.json"
        if not path.exists():
            raise KeyError(name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cast(dict[str, Any], payload)

    def compare(
        self, name: str, *, spot_id: str | None = None, bucket_key: str | None = None
    ) -> dict[str, Any]:
        return self.compare_dto(
            name=name, spot_id=spot_id, bucket_key=bucket_key
        ).to_dict()

    def compare_dto(
        self,
        name: str,
        *,
        spot_id: str | None = None,
        bucket_key: str | None = None,
    ) -> "ExternalSolverComparison":
        payload = self.load(name)
        capability_class = _normalize_capability_class(payload.get("capability_class"))
        for label in payload.get("labels", []):
            if spot_id is not None and label.get("spot_id") == spot_id:
                return ExternalSolverComparison(
                    matched_on="spot_id",
                    label=cast(dict[str, Any], label),
                    solver_name=str(payload.get("name", name)),
                    capability_class=capability_class,
                )
            if bucket_key is not None and label.get("bucket_key") == bucket_key:
                return ExternalSolverComparison(
                    matched_on="bucket_key",
                    label=cast(dict[str, Any], label),
                    solver_name=str(payload.get("name", name)),
                    capability_class=capability_class,
                )
        return ExternalSolverComparison(
            matched_on=None,
            label=None,
            solver_name=str(payload.get("name", name)),
            capability_class=capability_class,
        )


@dataclass(frozen=True)
class ExternalSolverCatalogEntry:
    name: str
    path: str
    label_count: int
    capability_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "label_count": self.label_count,
            "capability_class": self.capability_class,
        }


@dataclass(frozen=True)
class ExternalSolverComparison:
    matched_on: str | None
    label: dict[str, Any] | None
    solver_name: str
    capability_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_on": self.matched_on,
            "label": self.label,
            "solver_name": self.solver_name,
            "capability_class": self.capability_class,
        }
