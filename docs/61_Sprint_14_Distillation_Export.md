# Sprint 14 — Distillation / Export Readiness

Entregas implementadas:
- datasets exportados em `jsonl` prontos para consumo por treinos externos;
- linhas com `bucket_info`, `label_action`, `taxonomy_tags`, `split` e metadados de origem;
- manifestos que permitem rastrear geração e contagem.

Uso recomendado:
- treinos supervisionados fora do processo principal;
- avaliação comparativa entre `policy table`, `solver-like` e labels externos.
