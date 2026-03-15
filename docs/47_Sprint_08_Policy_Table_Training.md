# Sprint 08 — Policy table training local

Implementado:
- builder de dataset a partir de spot packs rotulados
- `PolicyTableTrainer`
- registry local em `var/models/`
- avaliação do modelo treinado contra os próprios spot packs

Objetivo:
- criar primeiro ciclo fechando `spot -> label -> train -> evaluate` sem dependência externa

Importante:
- isso ainda é um baseline tabular local, não substitui solver nem self-play forte.
