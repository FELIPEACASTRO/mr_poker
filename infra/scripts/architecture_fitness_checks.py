from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


LAYER_PATHS = {
    "apps": Path("apps"),
    "services": Path("services"),
    "packages": Path("packages"),
}

RULE_DESCRIPTIONS = {
    "R-001": "services nao podem importar apps (sem inversao de fronteira).",
    "R-002": "packages nao podem importar services nem apps.",
    "R-003": "services nao podem importar FastAPI/framework HTTP.",
}


@dataclass
class ImportRef:
    source_file: str
    source_module: str
    source_layer: str
    target_module: str
    target_layer: str | None
    line: int


def _as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def _iter_python_files(repo_root: Path) -> Iterable[tuple[str, Path]]:
    for layer_name, rel_root in LAYER_PATHS.items():
        root = repo_root / rel_root
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            yield layer_name, py_file


def _path_to_module(repo_root: Path, path: Path) -> str:
    rel = path.relative_to(repo_root).with_suffix("")
    return ".".join(rel.parts)


def _resolve_import_from_module(source_module: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    source_parts = source_module.split(".")
    package_parts = source_parts[:-1]
    keep_parts = len(package_parts) - (node.level - 1)
    if keep_parts < 0:
        keep_parts = 0
    base = package_parts[:keep_parts]
    if node.module:
        return ".".join(base + node.module.split("."))
    return ".".join(base)


def _layer_for_module(module_name: str) -> str | None:
    if not module_name:
        return None
    top = module_name.split(".", 1)[0]
    return top if top in LAYER_PATHS else None


def collect_import_refs(repo_root: Path) -> tuple[list[ImportRef], int]:
    refs: list[ImportRef] = []
    scanned_files = 0
    for source_layer, py_file in _iter_python_files(repo_root):
        scanned_files += 1
        source_module = _path_to_module(repo_root, py_file)
        rel_file = _as_posix(py_file.relative_to(repo_root))
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target_module = alias.name
                    refs.append(
                        ImportRef(
                            source_file=rel_file,
                            source_module=source_module,
                            source_layer=source_layer,
                            target_module=target_module,
                            target_layer=_layer_for_module(target_module),
                            line=node.lineno,
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                target_module = _resolve_import_from_module(source_module, node)
                refs.append(
                    ImportRef(
                        source_file=rel_file,
                        source_module=source_module,
                        source_layer=source_layer,
                        target_module=target_module,
                        target_layer=_layer_for_module(target_module),
                        line=node.lineno,
                    )
                )
    return refs, scanned_files


def _violation(rule_id: str, ref: ImportRef, detail: str) -> dict:
    return {
        "rule_id": rule_id,
        "detail": detail,
        "source_file": ref.source_file,
        "source_module": ref.source_module,
        "target_module": ref.target_module,
        "line": ref.line,
    }


def evaluate_rules(import_refs: list[ImportRef]) -> dict[str, list[dict]]:
    violations: dict[str, list[dict]] = {"R-001": [], "R-002": [], "R-003": []}
    for ref in import_refs:
        if ref.source_layer == "services" and ref.target_layer == "apps":
            violations["R-001"].append(
                _violation("R-001", ref, "services depende de apps")
            )
        if ref.source_layer == "packages" and ref.target_layer in {"services", "apps"}:
            violations["R-002"].append(
                _violation("R-002", ref, "packages depende de camada superior")
            )
        if ref.source_layer == "services" and (
            ref.target_module == "fastapi" or ref.target_module.startswith("fastapi.")
        ):
            violations["R-003"].append(
                _violation("R-003", ref, "services depende de FastAPI/HTTP framework")
            )
    return violations


def build_report(repo_root: Path) -> dict:
    import_refs, scanned_files = collect_import_refs(repo_root)
    violations = evaluate_rules(import_refs)
    total_violations = sum(len(items) for items in violations.values())
    rules = []
    for rule_id in sorted(RULE_DESCRIPTIONS):
        rule_violations = violations[rule_id]
        rules.append(
            {
                "rule_id": rule_id,
                "description": RULE_DESCRIPTIONS[rule_id],
                "status": "pass" if not rule_violations else "fail",
                "violations": rule_violations,
                "action": "nenhuma" if not rule_violations else "corrigir imports listados",
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": _as_posix(repo_root),
        "architecture_target": "monolito_modular_harden",
        "summary": {
            "scanned_files": scanned_files,
            "total_imports": len(import_refs),
            "total_violations": total_violations,
            "pass": total_violations == 0,
        },
        "rules": rules,
    }


def build_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# 102 - Architecture Conformance Report")
    lines.append("")
    lines.append("## Tipo")
    lines.append("")
    lines.append("- classificacao: auditoria")
    lines.append("- arquitetura alvo: `monolito_modular_harden`")
    lines.append("")
    lines.append("## Resumo")
    lines.append("")
    lines.append(f"- gerado em: `{report['generated_at']}`")
    lines.append(f"- arquivos analisados: `{report['summary']['scanned_files']}`")
    lines.append(f"- imports analisados: `{report['summary']['total_imports']}`")
    lines.append(f"- violacoes: `{report['summary']['total_violations']}`")
    lines.append(f"- status final: `{'pass' if report['summary']['pass'] else 'fail'}`")
    lines.append("")
    lines.append("## Matriz de Conformidade")
    lines.append("")
    lines.append("| Regra | Evidencia no codigo | Status | Acao |")
    lines.append("| --- | --- | --- | --- |")
    for rule in report["rules"]:
        if not rule["violations"]:
            evidence = "nenhuma violacao detectada"
        else:
            first = rule["violations"][0]
            evidence = f"{first['source_file']}:{first['line']} -> {first['target_module']}"
        lines.append(
            f"| `{rule['rule_id']}` {rule['description']} | `{evidence}` | `{rule['status']}` | `{rule['action']}` |"
        )

    lines.append("")
    lines.append("## Violacoes Detalhadas")
    lines.append("")
    has_violation = any(rule["violations"] for rule in report["rules"])
    if not has_violation:
        lines.append("- nenhuma violacao detectada.")
    else:
        for rule in report["rules"]:
            if not rule["violations"]:
                continue
            lines.append(f"### {rule['rule_id']}")
            lines.append("")
            for item in rule["violations"]:
                lines.append(
                    f"- `{item['source_file']}:{item['line']}` importa `{item['target_module']}` ({item['detail']})."
                )
            lines.append("")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run architecture fitness checks and generate a conformance report."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root path.")
    parser.add_argument(
        "--output-json",
        default="var/reports/architecture_conformance_report.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-md",
        default="docs/102_Architecture_Conformance_Report.md",
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when violations are present.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    if not output_json.is_absolute():
        output_json = repo_root / output_json
    if not output_md.is_absolute():
        output_md = repo_root / output_md

    report = build_report(repo_root)
    markdown = build_markdown(report)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    output_md.write_text(markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": _as_posix(output_json),
                "output_md": _as_posix(output_md),
                "summary": report["summary"],
            },
            indent=2,
        )
    )

    if args.strict and not report["summary"]["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
