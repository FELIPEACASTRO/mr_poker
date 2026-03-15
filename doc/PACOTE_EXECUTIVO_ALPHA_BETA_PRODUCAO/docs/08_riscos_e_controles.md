# Riscos e controles

| ID | Risco | Severidade | Sinal de alerta | Mitigação | Owner sugerido |
|---|---|---:|---|---|---|
| R1 | Dataset oficial mal congelado | Alta | divergência entre splits e runs | congelar manifesto, hash e owner | Dados |
| R2 | Candidate alpha fraco | Alta | round-robin inconsistente | ampliar labels, integrar solver real, reforçar benchmark | IA/ML |
| R3 | Coach incoerente | Média | explicação contradiz decisão | atrelar coach ao trace e taxonomy | IA/Produto |
| R4 | Custo local explode | Média | crescimento de artefatos e runs inúteis | budgets, retenção, limpeza automática | FinOps |
| R5 | Beta instável | Alta | incidentes repetidos no piloto | reforçar observabilidade, rollback e QA | MLOps/QA |
| R6 | Produção sem segurança mínima | Alta | segredos e auth improvisados | trilha dedicada de segurança antes do go-live | Segurança |
| R7 | Scope creep | Alta | backlog cresce sem gate | congelar escopo por fase | PO/Tech Lead |
| R8 | Release gate frouxo | Alta | promoção sem base objetiva | thresholds formais e sign-off | PO/IA/QA |