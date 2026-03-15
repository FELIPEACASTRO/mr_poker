# Sprint 02 — Persistência canônica, replay/regressão e baseline agent

## Objetivo
Sair de um engine POC puramente em memória para um fluxo reproduzível e regressivo.

## Entregas
- armazenamento local em SQLite para mãos, ações e snapshots
- replay determinístico a partir de seed/deck prefix + action log
- endpoint de replay
- baseline agent local, determinístico e explicável
- testes de regressão para roundtrip de persistência

## Limites desta sprint
- side pots complexos continuam fora do escopo
- baseline agent ainda não usa opponent modeling
- explicação continua curta e determinística
