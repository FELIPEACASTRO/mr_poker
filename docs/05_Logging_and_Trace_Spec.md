# Logging and Trace Spec

## Logs mínimos obrigatórios
- hand_id
- seed
- action log por evento
- snapshot log por transição relevante
- decision trace da IA
- resultado terminal da mão

## Uso
- depuração
- replay
- benchmark
- auditoria da decisão
- regressão

## Regra
Sem trace mínimo não há avanço de baseline para solver/data pipeline.
