# mr_poker Web UI

Mini-app React (Vite + TypeScript) com foco em navegacao rapida para fluxos de IA/ML e operacao poker local.

## Comandos

```bash
npm install
npm run dev
npm run build
npm run test:run
```

## Integracao

- A API serve o build em `GET /ui`.
- O fallback `GET /ui/{path:path}` suporta roteamento client-side.
- Variavel opcional: `VITE_API_BASE_URL` (default vazio para same-origin).
