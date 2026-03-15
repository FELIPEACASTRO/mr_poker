from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

INVENTORY_DOC = Path('docs/97_File_By_File_Inventory.md')
INTEGRATION_MAP_DOC = Path('docs/98_Data_AI_DB_Integration_Map.md')
API_ROOT_DIR = Path('apps/api')
CORE_DOSSIERS = [
    Path('docs/96_System_Knowledge_Dossier.md'),
    Path('docs/97_File_By_File_Inventory.md'),
    Path('docs/99_Gap_and_Risk_Register.md'),
    Path('docs/100_Multi_Specialist_Audit_and_Frontier_Roadmap.md'),
]

INVENTORY_ROW_PATTERN = re.compile(r'^\|\s*`([^`]+)`\s*\|')
DOC_ENDPOINT_PATTERN = re.compile(r'`(GET|POST|PUT|PATCH|DELETE)\s+(/[^`]*)`')
BASELINE_COUNT_PATTERN = re.compile(r'`?(\d+)`?\s+arquivos versionados')
STALE_BASELINE_PATTERN = re.compile(r'\b(307|312)\b')
HTTP_METHODS = {'get', 'post', 'put', 'patch', 'delete'}


def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def get_tracked_files(repo_root: Path) -> set[str]:
    output = subprocess.check_output(['git', 'ls-files'], cwd=repo_root, text=True)
    return {line.strip() for line in output.splitlines() if line.strip()}


def get_inventory_paths(path: Path) -> set[str]:
    paths = set()
    for line in _read_text(path).splitlines():
        match = INVENTORY_ROW_PATTERN.match(line)
        if match:
            paths.add(match.group(1))
    return paths


def _expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expr_name(node.value)
        return f'{parent}.{node.attr}' if parent else node.attr
    return ''


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_router_prefixes(tree: ast.Module) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(stmt.value, ast.Call):
            continue
        func_name = _expr_name(stmt.value.func)
        if not func_name.endswith('APIRouter'):
            continue
        prefix = ''
        for kw in stmt.value.keywords:
            if kw.arg == 'prefix':
                maybe_prefix = _literal_str(kw.value)
                if maybe_prefix is not None:
                    prefix = maybe_prefix
        prefixes[target.id] = prefix
    return prefixes


def get_api_routes(api_root: Path) -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for py_file in sorted(api_root.rglob('*.py')):
        if '__pycache__' in py_file.parts:
            continue
        tree = ast.parse(py_file.read_text(encoding='utf-8'))
        prefixes = _extract_router_prefixes(tree)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for deco in node.decorator_list:
                if (
                    not isinstance(deco, ast.Call)
                    or not isinstance(deco.func, ast.Attribute)
                    or deco.func.attr not in HTTP_METHODS
                ):
                    continue
                if not deco.args:
                    continue
                local_path = _literal_str(deco.args[0])
                if local_path is None:
                    continue
                owner_name = _expr_name(deco.func.value)
                prefix = prefixes.get(owner_name, '')
                full_path = f'{prefix}{local_path}' if prefix else local_path
                full_path = full_path.replace('//', '/')
                routes.add((deco.func.attr.upper(), full_path))
    return routes


def get_documented_endpoints(path: Path) -> set[tuple[str, str]]:
    endpoints = set()
    for method, endpoint in DOC_ENDPOINT_PATTERN.findall(_read_text(path)):
        endpoints.add((method.upper(), endpoint))
    return endpoints


def find_stale_count_mentions(path: Path) -> list[str]:
    return sorted(set(STALE_BASELINE_PATTERN.findall(_read_text(path))))


def find_mismatched_baseline_mentions(path: Path, expected_count: int) -> list[int]:
    mismatches = []
    for value in BASELINE_COUNT_PATTERN.findall(_read_text(path)):
        parsed = int(value)
        if parsed != expected_count:
            mismatches.append(parsed)
    return sorted(set(mismatches))


def _format_routes(routes: set[tuple[str, str]]) -> list[str]:
    return [f'{method} {endpoint}' for method, endpoint in sorted(routes)]


def run_consistency_audit(repo_root: Path) -> dict:
    tracked_files = get_tracked_files(repo_root)
    inventory_paths = get_inventory_paths(repo_root / INVENTORY_DOC)
    api_routes = get_api_routes(repo_root / API_ROOT_DIR)
    documented_endpoints = get_documented_endpoints(repo_root / INTEGRATION_MAP_DOC)

    stale_mentions = {}
    baseline_mismatches = {}
    for dossier in CORE_DOSSIERS:
        stale = find_stale_count_mentions(repo_root / dossier)
        if stale:
            stale_mentions[str(dossier)] = stale
        mismatches = find_mismatched_baseline_mentions(repo_root / dossier, len(tracked_files))
        if mismatches:
            baseline_mismatches[str(dossier)] = mismatches

    missing_inventory = tracked_files - inventory_paths
    extra_inventory = inventory_paths - tracked_files
    missing_endpoints = api_routes - documented_endpoints
    extra_endpoints = documented_endpoints - api_routes

    return {
        'tracked_file_count': len(tracked_files),
        'inventory_file_count': len(inventory_paths),
        'api_route_count': len(api_routes),
        'documented_endpoint_count': len(documented_endpoints),
        'missing_from_inventory': sorted(missing_inventory),
        'inventory_not_tracked': sorted(extra_inventory),
        'missing_endpoints_in_docs_98': _format_routes(missing_endpoints),
        'documented_endpoints_not_in_api': _format_routes(extra_endpoints),
        'stale_baseline_mentions': stale_mentions,
        'baseline_count_mismatches': baseline_mismatches,
    }


def has_hard_failures(report: dict) -> bool:
    return any(
        [
            report['missing_from_inventory'],
            report['missing_endpoints_in_docs_98'],
            report['documented_endpoints_not_in_api'],
            report['stale_baseline_mentions'],
            report['baseline_count_mismatches'],
        ]
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description='Validate repository audit consistency guards.')
    parser.add_argument('--repo-root', default='.', help='Repository root path.')
    parser.add_argument('--strict', action='store_true', help='Return non-zero when hard failures exist.')
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    report = run_consistency_audit(repo_root)
    print(json.dumps(report, indent=2))

    if args.strict and has_hard_failures(report):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
