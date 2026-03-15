# Business Requirements Document

## Visão
Criar uma plataforma local-first de Texas Hold'em Heads-Up que funcione como:
- sparring partner
- coach técnico
- replay analyzer
- base para evolução solver-centric

## Problema que resolve
Jogadores e desenvolvedores precisam de uma plataforma confiável para:
- jogar contra uma IA coerente
- revisar mãos
- estudar spots
- medir evolução
- experimentar estratégias em ambiente próprio

## Proposta de valor
- engine confiável
- replay reproduzível
- decisão explicável
- benchmark local
- aprendizado iterativo

## Personas
1. Jogador que quer treinar heads-up
2. Desenvolvedor de AI poker
3. Analista técnico que quer replay e traces
4. Futuro operador do produto em produção

## Objetivos de negócio
- reduzir risco técnico antes de investir em solver mais pesado
- construir base reutilizável para produto de treino
- validar arquitetura local-first
- preparar trilha até produção

## Não objetivos desta fase
- multiplayer completo
- solver de produção já resolvendo jogo inteiro
- operação em plataformas de terceiros

## KPIs iniciais
- engine correctness
- replay fidelity
- local session completion rate
- benchmark stability
- trace completeness
- hand/session persistence success rate

## Critérios de sucesso da fase atual
- jogar sessões locais fim a fim
- reproduzir mãos por seed/log
- gerar traces de decisão persistidos
- rodar benchmark e experimentos locais
- suportar spot packs para treino
