# Sprint 06 — What Changed

## Código
- `packages/taxonomy/spot_taxonomy.py`
- `packages/exports/phh_like.py`
- `services/taxonomy_service/service.py`
- `services/analytics_service/service.py`
- `services/export_service/service.py`
- `packages/persistence/sqlite_store.py` com `session_id` em `hands`
- `apps/api/main.py` com novos endpoints

## Testes
- `tests/unit/test_taxonomy_and_export.py`
- `tests/integration/test_api_taxonomy_export_analytics.py`

## Persistência
Agora cada mão pode ser vinculada a uma sessão, o que destrava:
- export por sessão
- analytics por sessão
- trilha de dados mais próxima do formato de dataset

## Valor de negócio
Esta sprint melhora:
- auditabilidade
- explicabilidade
- preparação para solver/datasets
- análise de treino do usuário
- organização do backlog técnico para a próxima fase
