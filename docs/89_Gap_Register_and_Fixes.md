# Registro de gaps e correções

## Gaps corrigidos
- instruções inconsistentes de execução local vs Docker
- ausência de healthcheck no compose
- ausência de `.dockerignore`
- ausência de runner explícito para integração
- ausência de runner explícito de validação completa
- persistência SQLite sem pragmas de robustez local

## Gaps ainda abertos
- integração com solver real
- policy/value distillation além de policy-table
- observabilidade de produção
- segurança e autenticação para ambiente multiusuário
- operação cloud com governança forte

## Decisão de produto mantida
O projeto continua **local-first** e a promoção correta é: Laboratório concluído -> Alpha controlado -> Beta fechado -> Produção inicial.
