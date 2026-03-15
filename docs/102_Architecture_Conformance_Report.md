# 102 - Architecture Conformance Report

## Tipo

- classificacao: auditoria
- arquitetura alvo: `monolito_modular_harden`

## Resumo

- gerado em: `2026-03-15T05:47:40.298321+00:00`
- arquivos analisados: `128`
- imports analisados: `415`
- violacoes: `0`
- status final: `pass`

## Matriz de Conformidade

| Regra | Evidencia no codigo | Status | Acao |
| --- | --- | --- | --- |
| `R-001` services nao podem importar apps (sem inversao de fronteira). | `nenhuma violacao detectada` | `pass` | `nenhuma` |
| `R-002` packages nao podem importar services nem apps. | `nenhuma violacao detectada` | `pass` | `nenhuma` |
| `R-003` services nao podem importar FastAPI/framework HTTP. | `nenhuma violacao detectada` | `pass` | `nenhuma` |

## Violacoes Detalhadas

- nenhuma violacao detectada.
