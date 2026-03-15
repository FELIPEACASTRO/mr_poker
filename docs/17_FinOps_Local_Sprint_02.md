# FinOps local — Sprint 02

## Regras
- persistência em SQLite local para custo zero de banco
- limites de experimentação por sessão
- sem LLM ou solver pesado no caminho crítico desta sprint
- replays e regressões devem rodar em CPU local

## Budget operacional local
- manter banco e snapshots pequenos
- armazenar apenas mão, action log e snapshots canônicos
- evitar duplicação de datasets pesados neste momento
