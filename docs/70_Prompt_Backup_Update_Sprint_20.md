# Prompt Backup — Sprint 20

Resumo do estado final desta etapa:
- solução local-first de Texas Hold'em NL Heads-Up;
- engine formal separado da camada de linguagem;
- traces persistidos e canônicos;
- taxonomy, analytics, coach e curriculum;
- policy table local e adapter para solver externo;
- readiness e packaging preparados.

Regras de ouro preservadas:
1. não usar LLM como policy final;
2. avaliar por H2H, exploitability proxies e estabilidade, não por “precisão 100%”;
3. separar claramente dados de trace, spot pack, solver-like e labels externos;
4. manter o projeto como laboratório/coach em ambiente próprio.
