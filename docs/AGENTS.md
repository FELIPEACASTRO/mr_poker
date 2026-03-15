# docs Scope Guide

## Escopo

`docs/*` contem especificacoes, historico de sprint, auditorias, ADRs, diagramas e runbooks.

## Classificacao documental

- Autoritativo: especificacoes correntes, ADRs, runbooks operacionais recentes.
- Historico: sprint previews, what changed, roadmaps e snapshots.
- Auditoria: auditorias, gap registers, checklists finais e relatorios de validacao.
- Executivo: o pacote em `doc/`, nao em `docs/`.

## Regras locais

- Preserve a numeracao ja adotada.
- Novos documentos devem declarar claramente se sao autoritativos, historicos, runbook ou auditoria.
- Nao use documentacao historica para contradizer comportamento do codigo atual sem registrar o gap.
- Quando uma conclusao depender de observacao do workspace e nao apenas do Git, explicite isso.

## Arquivos canonicos para onboarding

- `README.md`
- `docs/29_Business_Requirements_Document.md`
- `docs/30_Functional_Specification.md`
- `docs/31_Technical_Specification.md`
- `docs/32_AI_Data_Specification.md`
- `docs/94_Local_Execution_Runbook.md`
- `docs/96_System_Knowledge_Dossier.md`
- `docs/98_Data_AI_DB_Integration_Map.md`
- `docs/99_Gap_and_Risk_Register.md`

