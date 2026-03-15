# FinOps local — Sprint 01

## Objetivo
Controlar custo local e evitar desperdício antes da camada de treino.

## Política
- rodar tudo em CPU por padrão
- seeds fixas para reduzir reruns desnecessários
- testes curtos e determinísticos no pre-commit
- sem treinamento pesado nesta sprint
- logar tempo de execução dos testes principais

## Gate
A Sprint 01 só avança quando o engine estiver estável sem exigir hardware adicional.
