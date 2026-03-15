# infra Scope Guide

## Escopo

`infra/*` contem automacao operacional local e scripts de suporte.

## Regras locais

- Os scripts atuais sao orientados a ambiente Unix/bash.
- Em Windows/PowerShell, documente o equivalente em Python ou Docker antes de assumir execucao direta.
- Evite esconder pre-requisitos; se um script depende de `bash`, `pytest`, `ruff` ou Docker, declare isso de forma explicita.
- `healthcheck.py` e o artefato Python minimo e portavel desta pasta.

## Responsabilidades

- Bootstrap local.
- Limpeza/sync do repositorio.
- Execucao de suites unitarias, integracao e validacao completa.
- Healthcheck simples de API local.

## Limites atuais

- Nao ha infraestrutura cloud real nesta pasta.
- Nao ha pipeline de deploy produtivo; o foco e automacao local e container local.

