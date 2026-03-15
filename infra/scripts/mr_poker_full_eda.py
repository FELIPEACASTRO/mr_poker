from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
PATH_PARAM_PATTERN = re.compile(r"\{([^{}]+)\}")
CLIENT_CALL_PATTERN = re.compile(r"client\.(get|post|put|patch|delete)\(\s*f?['\"]([^'\"]+)['\"]")
ASSERT_IN_PATTERN = re.compile(r"assert\s+['\"]([^'\"]+)['\"]\s+in\s+")

TARGET_SERVICES_FOR_METRICS = {
    "ReadinessService": "snapshot",
    "EvaluationService": "evaluate_model",
    "CalibrationService": "calibrate",
    "GovernanceService": "build_model_card",
    "CurriculumService": "build_for_session",
}

DATA_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv", ".db", ".sqlite"}
MAX_DEEP_ANALYSIS_BYTES = 10 * 1024 * 1024
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _safe_unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expr_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return _safe_unparse(node) or "<unknown>"


def _is_basemodel_subclass(class_node: ast.ClassDef) -> bool:
    for base in class_node.bases:
        name = _expr_name(base)
        if name.endswith("BaseModel"):
            return True
    return False


def _is_dataclass(class_node: ast.ClassDef) -> bool:
    for deco in class_node.decorator_list:
        if isinstance(deco, ast.Name) and deco.id == "dataclass":
            return True
        if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Name) and deco.func.id == "dataclass":
            return True
    return False


def _extract_class_fields(class_node: ast.ClassDef) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for stmt in class_node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields.append(
                {
                    "name": stmt.target.id,
                    "annotation": _safe_unparse(stmt.annotation),
                    "default": _safe_unparse(stmt.value),
                    "line": stmt.lineno,
                }
            )
    return fields


def _load_python_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_eda_module(repo_root: Path, explicit_script: Path | None):
    candidates: list[Path] = []
    if explicit_script is not None:
        candidates.append(explicit_script)
    candidates.append(
        Path.home()
        / ".agents"
        / "skills"
        / "K-Dense-AI__claude-scientific-skills"
        / "scientific-skills"
        / "exploratory-data-analysis"
        / "scripts"
        / "eda_analyzer.py"
    )
    candidates.append(
        repo_root
        / ".agents"
        / "skills"
        / "K-Dense-AI__claude-scientific-skills"
        / "scientific-skills"
        / "exploratory-data-analysis"
        / "scripts"
        / "eda_analyzer.py"
    )
    for path in candidates:
        if path.exists():
            module = _load_python_module(path, "mr_poker_skill_eda_analyzer")
            if module is not None:
                return module, path
    return None, None


def _get_tracked_files(repo_root: Path) -> set[str]:
    output = subprocess.check_output(["git", "ls-files"], cwd=repo_root, text=True)
    return {line.strip() for line in output.splitlines() if line.strip()}


def _normalize_path_template(path: str) -> str:
    return PATH_PARAM_PATTERN.sub("{param}", path)


def _extract_return_contract(function_node: FunctionNode) -> dict[str, Any]:
    dict_assignments: dict[str, set[str]] = {}
    explicit_keys: set[str] = set()
    dynamic_returns: list[str] = []

    for stmt in function_node.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Dict)
        ):
            keys = {
                key.value
                for key in stmt.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if keys:
                dict_assignments[stmt.targets[0].id] = keys

        if isinstance(stmt, ast.Return):
            value = stmt.value
            if isinstance(value, ast.Dict):
                for key in value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        explicit_keys.add(key.value)
                continue

            if isinstance(value, ast.Name) and value.id in dict_assignments:
                explicit_keys.update(dict_assignments[value.id])
                continue

            dynamic_returns.append(_safe_unparse(value) or "<dynamic>")

    return {
        "explicit_keys": sorted(explicit_keys),
        "dynamic_returns": dynamic_returns,
        "has_dynamic_return": bool(dynamic_returns),
    }


def _collect_owner_calls(function_node: FunctionNode, owner_map: dict[str, str]) -> list[dict[str, Any]]:
    calls: list[tuple[int, str, str, str]] = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        var_name = node.func.value.id
        if var_name == "app":
            continue
        if var_name not in owner_map:
            continue
        calls.append((node.lineno, owner_map[var_name], var_name, node.func.attr))

    calls.sort(key=lambda item: item[0])
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for line, owner, variable, method in calls:
        key = (owner, variable, method)
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "owner": owner,
                "variable": variable,
                "method": method,
                "line": line,
            }
        )
    return unique


def _iter_api_python_files(repo_root: Path) -> list[Path]:
    api_root = repo_root / "apps" / "api"
    files: list[Path] = []
    for path in sorted(api_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files.append(path)
    return files


def _extract_container_owner_map(repo_root: Path) -> dict[str, str]:
    container_path = repo_root / "apps" / "api" / "container.py"
    if not container_path.exists():
        return {}
    tree = ast.parse(container_path.read_text(encoding="utf-8"))
    owner_map: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "AppContainer":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            annotation = _safe_unparse(stmt.annotation) or "UnknownOwner"
            owner_map[stmt.target.id] = annotation.split(".")[-1]
    return owner_map


def _extract_router_prefixes(tree: ast.Module) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(stmt.value, ast.Call):
            continue
        func_name = _expr_name(stmt.value.func)
        if not func_name.endswith("APIRouter"):
            continue
        prefix = ""
        for kw in stmt.value.keywords:
            if kw.arg == "prefix":
                maybe_prefix = _literal_str(kw.value)
                if maybe_prefix is not None:
                    prefix = maybe_prefix
        prefixes[target.id] = prefix
    return prefixes


def _collect_owner_calls_for_container(
    function_node: FunctionNode, container_owner_map: dict[str, str]
) -> list[dict[str, Any]]:
    calls: list[tuple[int, str, str, str]] = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        base = node.func.value
        if (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.value.id == "container"
        ):
            field = base.attr
            owner = container_owner_map.get(field, field)
            calls.append((node.lineno, owner, field, node.func.attr))
        elif isinstance(base, ast.Name) and base.id in container_owner_map:
            owner = container_owner_map[base.id]
            calls.append((node.lineno, owner, base.id, node.func.attr))

    calls.sort(key=lambda item: item[0])
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for line, owner, variable, method in calls:
        key = (owner, variable, method)
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "owner": owner,
                "variable": variable,
                "method": method,
                "line": line,
            }
        )
    return unique


def _extract_api_contracts(repo_root: Path, request_model_names: set[str]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    container_owner_map = _extract_container_owner_map(repo_root)
    for path in _iter_api_python_files(repo_root):
        rel_file = str(path.relative_to(repo_root)).replace("\\", "/")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        router_prefixes = _extract_router_prefixes(tree)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            route_decorators: list[tuple[ast.Call, str]] = []
            for deco in node.decorator_list:
                if (
                    isinstance(deco, ast.Call)
                    and isinstance(deco.func, ast.Attribute)
                    and isinstance(deco.func.value, ast.Name)
                    and deco.func.attr in HTTP_METHODS
                ):
                    route_decorators.append((deco, deco.func.value.id))
            if not route_decorators:
                continue

            return_contract = _extract_return_contract(node)
            owner_calls = _collect_owner_calls_for_container(node, container_owner_map)
            owner_primary = owner_calls[0]["owner"] if owner_calls else "API Router"

            for deco, router_name in route_decorators:
                local_path = _literal_str(deco.args[0]) if deco.args else None
                if local_path is None:
                    local_path = _safe_unparse(deco.args[0]) if deco.args else "<dynamic>"
                prefix = router_prefixes.get(router_name, "")
                full_path = f"{prefix}{local_path}" if prefix else local_path
                full_path = full_path.replace("//", "/")
                path_params = sorted(set(PATH_PARAM_PATTERN.findall(full_path)))

                args = node.args.args
                defaults = [None] * (len(args) - len(node.args.defaults)) + list(node.args.defaults)
                query_params: list[dict[str, Any]] = []
                body_params: list[dict[str, Any]] = []
                path_params_contract: list[dict[str, Any]] = []

                for arg_node, default_node in zip(args, defaults):
                    annotation = _safe_unparse(arg_node.annotation)
                    if annotation == "Request":
                        continue
                    parameter = {
                        "name": arg_node.arg,
                        "annotation": annotation,
                        "default": _safe_unparse(default_node),
                        "line": arg_node.lineno,
                    }
                    if arg_node.arg in path_params:
                        path_params_contract.append(parameter)
                    elif annotation in request_model_names:
                        body_params.append(parameter)
                    else:
                        query_params.append(parameter)

                contracts.append(
                    {
                        "method": deco.func.attr.upper(),
                        "path": full_path,
                        "handler": node.name,
                        "owner_primary": owner_primary,
                        "owners": sorted({item["owner"] for item in owner_calls}),
                        "owner_call_evidence": owner_calls,
                        "path_params": path_params_contract,
                        "query_params": query_params,
                        "body_params": body_params,
                        "response_contract": {
                            "explicit_keys": return_contract["explicit_keys"],
                            "dynamic_returns": return_contract["dynamic_returns"],
                            "has_dynamic_return": return_contract["has_dynamic_return"],
                            "derived_from_tests": False,
                            "derived_keys_from_tests": [],
                            "test_evidence": [],
                        },
                        "source": {
                            "file": rel_file,
                            "line": node.lineno,
                        },
                    }
                )

    contracts.sort(key=lambda item: (item["method"], item["path"]))
    deduped: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for contract in contracts:
        pair = (contract["method"], contract["path"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped.append(contract)
    return deduped


def _extract_request_models(repo_root: Path) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for path in _iter_api_python_files(repo_root):
        rel_file = str(path.relative_to(repo_root)).replace("\\", "/")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and _is_basemodel_subclass(node):
                fields = _extract_class_fields(node)
                models.append(
                    {
                        "name": node.name,
                        "kind": "BaseModel",
                        "fields": fields,
                        "source": {
                            "file": rel_file,
                            "line": node.lineno,
                        },
                    }
                )
    models.sort(key=lambda item: item["name"])
    return models


def _extract_service_io_contracts(repo_root: Path) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for service_path in sorted((repo_root / "services").glob("*/service.py")):
        tree = ast.parse(service_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not node.name.endswith("Service"):
                continue

            methods: list[dict[str, Any]] = []
            for method in node.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if method.name.startswith("_"):
                    continue

                args = method.args.args
                defaults = [None] * (len(args) - len(method.args.defaults)) + list(method.args.defaults)
                params: list[dict[str, Any]] = []
                for arg_node, default_node in zip(args, defaults):
                    if arg_node.arg == "self":
                        continue
                    params.append(
                        {
                            "name": arg_node.arg,
                            "annotation": _safe_unparse(arg_node.annotation),
                            "default": _safe_unparse(default_node),
                            "line": arg_node.lineno,
                        }
                    )

                return_contract = _extract_return_contract(method)
                methods.append(
                    {
                        "name": method.name,
                        "params": params,
                        "return_annotation": _safe_unparse(method.returns),
                        "returns": {
                            "explicit_keys": return_contract["explicit_keys"],
                            "dynamic_returns": return_contract["dynamic_returns"],
                            "has_dynamic_return": return_contract["has_dynamic_return"],
                        },
                        "source": {
                            "file": str(service_path.relative_to(repo_root)).replace("\\", "/"),
                            "line": method.lineno,
                        },
                    }
                )

            contracts.append(
                {
                    "service": node.name,
                    "source": {
                        "file": str(service_path.relative_to(repo_root)).replace("\\", "/"),
                        "line": node.lineno,
                    },
                    "methods": methods,
                }
            )
    return contracts


def _extract_dataclass_fields(repo_root: Path, relative_path: str, class_name: str) -> list[dict[str, Any]]:
    path = repo_root / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name and (_is_dataclass(node) or _is_basemodel_subclass(node)):
            return _extract_class_fields(node)
    return []


def _extract_dict_return_keys(path: Path, function_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            contract = _extract_return_contract(node)
            return contract["explicit_keys"]
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        for method in class_node.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name == function_name:
                contract = _extract_return_contract(method)
                return contract["explicit_keys"]
    return []


def _extract_feature_catalog(repo_root: Path, service_contracts: list[dict[str, Any]]) -> dict[str, Any]:
    minimum_fields = _extract_dataclass_fields(
        repo_root, "packages/features/minimum.py", "MinimumDecisionFeatures"
    )
    baseline_fields = _extract_dataclass_fields(
        repo_root, "packages/features/poker.py", "BaselineFeatures"
    )

    buckets_path = repo_root / "packages/solver_like/buckets.py"
    normalize_path = repo_root / "packages/solver_like/normalize.py"
    labeler_path = repo_root / "packages/solver_like/labeler.py"

    buckets_tree = ast.parse(buckets_path.read_text(encoding="utf-8"))
    bucket_functions = [
        {"name": node.name, "line": node.lineno}
        for node in buckets_tree.body
        if isinstance(node, ast.FunctionDef) and "bucket" in node.name
    ]

    solver_bucket_keys = _extract_dict_return_keys(buckets_path, "bucketize_spot")
    normalize_runtime_keys = _extract_dict_return_keys(normalize_path, "normalize_runtime")
    normalize_snapshot_keys = _extract_dict_return_keys(normalize_path, "normalize_snapshot_trace")
    labeler_keys = _extract_dict_return_keys(labeler_path, "label_spot")

    service_metrics: dict[str, Any] = {}
    for contract in service_contracts:
        service_name = contract["service"]
        if service_name not in TARGET_SERVICES_FOR_METRICS:
            continue
        method_name = TARGET_SERVICES_FOR_METRICS[service_name]
        method_contract = next(
            (method for method in contract["methods"] if method["name"] == method_name),
            None,
        )
        if method_contract is None:
            continue
        service_metrics[service_name] = {
            "method": method_name,
            "explicit_keys": method_contract["returns"]["explicit_keys"],
            "source": method_contract["source"],
        }

    required_blocks = {"minimum", "baseline", "solver-like"}
    blocks_present = set()
    if minimum_fields:
        blocks_present.add("minimum")
    if baseline_fields:
        blocks_present.add("baseline")
    if solver_bucket_keys or normalize_runtime_keys or labeler_keys:
        blocks_present.add("solver-like")

    return {
        "decision_features": {
            "minimum": {
                "fields": minimum_fields,
                "source": "packages/features/minimum.py",
            },
            "baseline": {
                "fields": baseline_fields,
                "source": "packages/features/poker.py",
            },
        },
        "solver_like": {
            "bucket_functions": bucket_functions,
            "bucket_tags": solver_bucket_keys,
            "normalize_runtime_fields": normalize_runtime_keys,
            "normalize_snapshot_fields": normalize_snapshot_keys,
            "label_output_fields": labeler_keys,
            "sources": [
                "packages/solver_like/buckets.py",
                "packages/solver_like/normalize.py",
                "packages/solver_like/labeler.py",
            ],
        },
        "service_metrics": service_metrics,
        "coverage": {
            "required_blocks": sorted(required_blocks),
            "blocks_present": sorted(blocks_present),
            "blocks_missing": sorted(required_blocks - blocks_present),
        },
    }


def _summarize_data_analysis(data_analysis: Any) -> dict[str, Any]:
    if not isinstance(data_analysis, dict):
        return {}
    summary: dict[str, Any] = {}
    for key, value in data_analysis.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, dict):
            summary[f"{key}_keys"] = sorted(value.keys())[:20]
            summary[f"{key}_size"] = len(value)
        elif isinstance(value, list):
            summary[f"{key}_count"] = len(value)
        else:
            summary[key] = str(type(value).__name__)
    return summary


def _collect_var_files(repo_root: Path, include_var_runtime: bool) -> list[Path]:
    var_root = repo_root / "var"
    if not var_root.exists():
        return []

    if include_var_runtime:
        candidates = [path for path in var_root.rglob("*") if path.is_file()]
    else:
        tracked = _get_tracked_files(repo_root)
        candidates = []
        for rel in sorted(tracked):
            if not rel.startswith("var/"):
                continue
            path = repo_root / rel
            if path.exists() and path.is_file():
                candidates.append(path)

    files: list[Path] = []
    for path in sorted(candidates):
        if path.name == ".gitkeep":
            continue
        if path.suffix.lower() in DATA_EXTENSIONS:
            files.append(path)
    return files


def _analyze_data_file(path: Path, eda_module) -> dict[str, Any]:
    item = {
        "path": str(path).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "extension": path.suffix.lower(),
    }
    if eda_module is None:
        item["analysis"] = {
            "status": "skill_unavailable",
            "message": "EDA analyzer module was not found.",
        }
        return item

    if path.stat().st_size > MAX_DEEP_ANALYSIS_BYTES:
        item["analysis"] = {
            "status": "skipped",
            "reason": f"file larger than {MAX_DEEP_ANALYSIS_BYTES} bytes",
        }
        return item

    try:
        analysis = eda_module.analyze_file(str(path))
        file_type = analysis.get("file_type", {})
        data_analysis = analysis.get("data_analysis", {})
        item["analysis"] = {
            "status": "ok",
            "category": file_type.get("category"),
            "description": file_type.get("description"),
            "summary": _summarize_data_analysis(data_analysis),
        }
    except Exception as exc:
        item["analysis"] = {"status": "error", "message": str(exc)}
    return item


def _extract_data_artifacts(
    repo_root: Path, include_var_runtime: bool, eda_module
) -> dict[str, Any]:
    files = _collect_var_files(repo_root, include_var_runtime)
    items = [_analyze_data_file(path, eda_module) for path in files]

    by_extension: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for item in items:
        ext = item["extension"] or "<none>"
        by_extension[ext] = by_extension.get(ext, 0) + 1
        category = item.get("analysis", {}).get("category", "unknown")
        by_category[category] = by_category.get(category, 0) + 1

    return {
        "include_var_runtime": include_var_runtime,
        "analyzed_file_count": len(items),
        "by_extension": dict(sorted(by_extension.items())),
        "by_category": dict(sorted(by_category.items())),
        "files": items,
    }


def _extract_test_evidence(repo_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for test_file in sorted((repo_root / "tests").rglob("test_*.py")):
        text = test_file.read_text(encoding="utf-8")
        lines = text.splitlines()
        file_asserted_keys = sorted(set(ASSERT_IN_PATTERN.findall(text)))
        rel_file = str(test_file.relative_to(repo_root)).replace("\\", "/")
        for idx, line in enumerate(lines, start=1):
            match = CLIENT_CALL_PATTERN.search(line)
            if not match:
                continue
            method = match.group(1).upper()
            path = match.group(2)
            key = (method, _normalize_path_template(path))
            existing = evidence.get(key, {"calls": [], "asserted_keys": []})
            existing["calls"].append({"file": rel_file, "line": idx, "path_expr": path})
            merged_keys = sorted(set(existing["asserted_keys"]) | set(file_asserted_keys))
            existing["asserted_keys"] = merged_keys
            evidence[key] = existing
    return evidence


def _attach_test_evidence(
    api_contracts: list[dict[str, Any]],
    test_evidence: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for contract in api_contracts:
        key = (contract["method"], _normalize_path_template(contract["path"]))
        evidence = test_evidence.get(key)
        if evidence is None:
            continue
        response_contract = contract["response_contract"]
        response_contract["test_evidence"] = evidence["calls"]
        if not response_contract["explicit_keys"]:
            response_contract["derived_from_tests"] = True
            response_contract["derived_keys_from_tests"] = evidence["asserted_keys"]


def _compute_coverage_and_gaps(
    api_contracts: list[dict[str, Any]],
    request_models: list[dict[str, Any]],
    feature_catalog: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    api_routes_total = len(api_contracts)
    api_routes_covered = sum(1 for _ in api_contracts)

    request_models_total = len(request_models)
    request_models_covered = sum(1 for model in request_models if model["fields"])

    required_blocks = feature_catalog["coverage"]["required_blocks"]
    blocks_missing = feature_catalog["coverage"]["blocks_missing"]
    feature_blocks_total = len(required_blocks)
    feature_blocks_covered = feature_blocks_total - len(blocks_missing)

    gaps: list[dict[str, Any]] = []

    has_review_endpoint = any(
        contract["method"] == "GET" and contract["path"] == "/v1/hands/{hand_id}/review"
        for contract in api_contracts
    )
    if not has_review_endpoint:
        gaps.append(
            {
                "category": "api",
                "severity": "high",
                "message": "Mandatory endpoint GET /v1/hands/{hand_id}/review not found.",
            }
        )

    if request_models_covered != request_models_total:
        missing = [model["name"] for model in request_models if not model["fields"]]
        gaps.append(
            {
                "category": "request_models",
                "severity": "high",
                "message": "Some request models were not fully covered.",
                "missing_models": missing,
            }
        )

    if blocks_missing:
        gaps.append(
            {
                "category": "feature_blocks",
                "severity": "high",
                "message": "Required feature blocks are missing.",
                "missing_blocks": blocks_missing,
            }
        )

    coverage = {
        "api_routes_total": api_routes_total,
        "api_routes_covered": api_routes_covered,
        "request_models_total": request_models_total,
        "request_models_covered": request_models_covered,
        "feature_blocks_total": feature_blocks_total,
        "feature_blocks_covered": feature_blocks_covered,
    }
    return coverage, gaps


def build_report(
    repo_root: Path,
    include_var_runtime: bool = True,
    eda_skill_script: Path | None = None,
) -> dict[str, Any]:
    request_models = _extract_request_models(repo_root)
    request_model_names = {model["name"] for model in request_models}
    api_contracts = _extract_api_contracts(repo_root, request_model_names)
    service_io_contracts = _extract_service_io_contracts(repo_root)
    feature_catalog = _extract_feature_catalog(repo_root, service_io_contracts)

    test_evidence = _extract_test_evidence(repo_root)
    _attach_test_evidence(api_contracts, test_evidence)

    eda_module, eda_module_path = _find_eda_module(repo_root, eda_skill_script)
    data_artifacts = _extract_data_artifacts(repo_root, include_var_runtime, eda_module)
    if eda_module_path is not None:
        data_artifacts["eda_module_path"] = str(eda_module_path).replace("\\", "/")
    else:
        data_artifacts["eda_module_path"] = None

    coverage, gaps = _compute_coverage_and_gaps(api_contracts, request_models, feature_catalog)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root).replace("\\", "/"),
        "api_contracts": api_contracts,
        "api_request_models": request_models,
        "service_io_contracts": service_io_contracts,
        "feature_catalog": feature_catalog,
        "data_artifacts": data_artifacts,
        "coverage": coverage,
        "gaps": gaps,
    }
    return report


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 101 - Exploratory Data Analysis Full I/O and Features")
    lines.append("")
    lines.append("## Resumo")
    lines.append("")
    lines.append(f"- Gerado em: `{payload['generated_at']}`")
    lines.append(f"- Repo root: `{payload['repo_root']}`")
    lines.append(f"- Rotas API mapeadas: `{payload['coverage']['api_routes_covered']}/{payload['coverage']['api_routes_total']}`")
    lines.append(
        f"- Request models mapeados: `{payload['coverage']['request_models_covered']}/{payload['coverage']['request_models_total']}`"
    )
    lines.append(
        f"- Blocos de feature mapeados: `{payload['coverage']['feature_blocks_covered']}/{payload['coverage']['feature_blocks_total']}`"
    )
    lines.append("")
    lines.append("## API Contracts")
    lines.append("")
    lines.append("| Metodo | Path | Owner primario | Fonte |")
    lines.append("| --- | --- | --- | --- |")
    for contract in payload["api_contracts"]:
        source = contract["source"]
        lines.append(
            f"| `{contract['method']}` | `{contract['path']}` | `{contract['owner_primary']}` | `{source['file']}:{source['line']}` |"
        )

    lines.append("")
    lines.append("## Request Models (BaseModel)")
    lines.append("")
    for model in payload["api_request_models"]:
        source = model["source"]
        lines.append(f"### `{model['name']}` ({source['file']}:{source['line']})")
        lines.append("")
        lines.append("| Campo | Tipo | Default |")
        lines.append("| --- | --- | --- |")
        for field in model["fields"]:
            default = field["default"] or "-"
            lines.append(f"| `{field['name']}` | `{field['annotation']}` | `{default}` |")
        lines.append("")

    lines.append("## Feature Catalog")
    lines.append("")

    lines.append("### Decision Features - Minimum")
    lines.append("")
    for field in payload["feature_catalog"]["decision_features"]["minimum"]["fields"]:
        lines.append(f"- `{field['name']}`: `{field['annotation']}`")
    lines.append("")

    lines.append("### Decision Features - Baseline")
    lines.append("")
    for field in payload["feature_catalog"]["decision_features"]["baseline"]["fields"]:
        lines.append(f"- `{field['name']}`: `{field['annotation']}`")
    lines.append("")

    lines.append("### Solver-like")
    lines.append("")
    solver = payload["feature_catalog"]["solver_like"]
    lines.append(f"- Bucket functions: `{', '.join(item['name'] for item in solver['bucket_functions'])}`")
    lines.append(f"- Bucket tags: `{', '.join(solver['bucket_tags'])}`")
    lines.append(f"- normalize_runtime fields: `{', '.join(solver['normalize_runtime_fields'])}`")
    lines.append(f"- normalize_snapshot fields: `{', '.join(solver['normalize_snapshot_fields'])}`")
    lines.append(f"- label output fields: `{', '.join(solver['label_output_fields'])}`")
    lines.append("")

    lines.append("### Service Metrics")
    lines.append("")
    for service_name, service_data in payload["feature_catalog"]["service_metrics"].items():
        source = service_data["source"]
        keys = ", ".join(service_data["explicit_keys"])
        lines.append(f"- `{service_name}.{service_data['method']}` (`{source['file']}:{source['line']}`): `{keys}`")
    lines.append("")

    lines.append("## Data Artifacts (var/*)")
    lines.append("")
    data_artifacts = payload["data_artifacts"]
    lines.append(f"- include_var_runtime: `{data_artifacts['include_var_runtime']}`")
    lines.append(f"- EDA module path: `{data_artifacts['eda_module_path']}`")
    lines.append(f"- Arquivos analisados: `{data_artifacts['analyzed_file_count']}`")
    lines.append("")
    lines.append("### Distribuicao por extensao")
    lines.append("")
    lines.append("| Extensao | Count |")
    lines.append("| --- | ---: |")
    for ext, count in data_artifacts["by_extension"].items():
        lines.append(f"| `{ext}` | `{count}` |")
    lines.append("")
    lines.append("### Distribuicao por categoria EDA")
    lines.append("")
    lines.append("| Categoria | Count |")
    lines.append("| --- | ---: |")
    for category, count in data_artifacts["by_category"].items():
        lines.append(f"| `{category}` | `{count}` |")
    lines.append("")

    lines.append("## Gaps")
    lines.append("")
    if payload["gaps"]:
        for gap in payload["gaps"]:
            lines.append(f"- `{gap['category']}` [{gap['severity']}]: {gap['message']}")
    else:
        lines.append("- Nenhum gap P0 detectado para API/request models/feature blocks obrigatorios.")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a full code-level EDA inventory for mr_poker (API I/O + features + var artifacts)."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root path.")
    parser.add_argument(
        "--output-json",
        default="var/reports/mr_poker_full_io_features.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-md",
        default="docs/101_Exploratory_Data_Analysis_Full_IO_Features.md",
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--include-var-runtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include observed runtime files under var/ (default: true).",
    )
    parser.add_argument(
        "--eda-skill-script",
        default=None,
        help="Optional explicit path to exploratory-data-analysis/scripts/eda_analyzer.py.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = repo_root / output_json
    output_md = Path(args.output_md)
    if not output_md.is_absolute():
        output_md = repo_root / output_md

    eda_skill_script = Path(args.eda_skill_script) if args.eda_skill_script else None
    if eda_skill_script is not None and not eda_skill_script.is_absolute():
        eda_skill_script = (repo_root / eda_skill_script).resolve()

    payload = build_report(
        repo_root=repo_root,
        include_var_runtime=args.include_var_runtime,
        eda_skill_script=eda_skill_script,
    )
    markdown = build_markdown_report(payload)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    output_md.write_text(markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(output_json).replace("\\", "/"),
                "output_md": str(output_md).replace("\\", "/"),
                "coverage": payload["coverage"],
                "gaps": payload["gaps"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
