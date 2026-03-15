```mermaid
flowchart LR
    T[Decision Traces] --> B[Dataset Builder]
    S[Spot Packs + Solver-like] --> B
    B --> D[(Dataset JSONL + Manifest)]
    D --> E[Evaluation by Split/Taxonomy]
    E --> R[Reports]
```
