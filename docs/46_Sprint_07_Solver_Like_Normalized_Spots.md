# Sprint 07 — Solver-like normalized spots

Implementado nesta sprint:
- normalização explícita de spots ativos por mão
- bucketização inicial por street/posição/SPR/equity/preço/texture
- labeler `solver_like_v1_rule` para rotular ações de referência local
- comparação `baseline vs solver-like` em spot packs

O que isso resolve:
- cria trilha de dados intermediária antes do solver real
- permite medir alinhamento do baseline em spots canônicos
- prepara dataset local para distilação

Limite atual:
- não é solver real; é rotulagem heurística reproduzível e auditável.
