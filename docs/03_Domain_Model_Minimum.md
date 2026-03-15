# Domain Model — Minimum

## Entidades mínimas
- Card
- PlayerState
- ActionEvent
- HandState
- ActionLog
- SnapshotLog
- DecisionTrace

## Invariantes mínimas
- deck com 52 cartas únicas
- cada jogador com 2 hole cards no heads-up
- board com no máximo 5 cartas
- pot >= 0
- to_call >= 0
- street sempre válida
- actor_seat precisa existir na mão

## Regra arquitetural
Toda feature, explicação e decisão deve depender de um estado válido do engine.
