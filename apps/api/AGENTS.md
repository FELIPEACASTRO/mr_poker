# apps/api Scope Guide

## Escopo

Esta pasta contem a superficie HTTP do sistema. No estado versionado atual, o arquivo relevante e `main.py`.

## Responsabilidade

- Instanciar `FastAPI` e compor o runtime com `GameEngine`, `SqliteHandStore` e `services/*`.
- Definir os contratos Pydantic de entrada.
- Traduzir excecoes de dominio para `HTTPException`.
- Expor endpoints sem mover logica de negocio pesada para a camada de rota.

## Regras locais

- Novas rotas devem ter owner explicito em `services/*`.
- Persistencia direta em rota so e aceitavel para a trilha minima de hand runtime (`new`, `actions`, `auto`, `get hand`); fora disso, prefira service.
- Preserve compatibilidade dos payloads ja usados pelos testes de integracao.
- Qualquer endpoint novo precisa ser refletido no dossie `docs/96_*` e no mapa `docs/98_*`.

## Validacao minima

- `python -m pytest -q`
- Priorize os testes em `tests/integration/*` ao tocar esta pasta.

