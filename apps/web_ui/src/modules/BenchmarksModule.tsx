import { useState } from "react";

import { apiRequest } from "../lib/httpClient";
import type { ApiJson } from "../lib/types";

type BenchmarkConfig = {
  title: string;
  endpoint: string;
  payload: ApiJson;
};

const BENCHMARKS: BenchmarkConfig[] = [
  {
    title: "Smoke Benchmark",
    endpoint: "/v1/benchmark/smoke",
    payload: { num_hands: 40, seed_base: 11111, stacks: [100, 100] },
  },
  {
    title: "Adaptive Benchmark",
    endpoint: "/v1/benchmark/adaptive",
    payload: { num_hands: 40, seed_base: 22222, stacks: [100, 100] },
  },
  {
    title: "H2H Benchmark",
    endpoint: "/v1/benchmark/h2h",
    payload: { num_matches: 2, hands_per_match: 20, seed_base: 33333, stacks: [100, 100] },
  },
  {
    title: "Torneio Round-Robin",
    endpoint: "/v1/tournaments/round-robin",
    payload: { entrants: ["baseline", "adaptive"], hands_per_match: 10, seed_base: 44444, stacks: [100, 100] },
  },
];

export function BenchmarksModule() {
  const [selectedResult, setSelectedResult] = useState<ApiJson | null>(null);
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function runBenchmark(config: BenchmarkConfig) {
    setLoadingKey(config.endpoint);
    setError(null);
    try {
      const result = await apiRequest<ApiJson>(config.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config.payload),
      });
      setSelectedResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao executar benchmark.");
    } finally {
      setLoadingKey(null);
    }
  }

  return (
    <section className="module-grid">
      <div className="hero-card">
        <div className="hero-chip">♥ Benchmarks</div>
        <h1>Benchmarks e Torneios</h1>
        <p>Execução rápida de baterias padrão para avaliar baseline/adaptive no laboratório local.</p>
      </div>

      {error ? <p className="error-box">{error}</p> : null}

      <div className="card-grid">
        {BENCHMARKS.map((benchmark) => (
          <article className="data-card" key={benchmark.endpoint}>
            <h2>{benchmark.title}</h2>
            <pre>{JSON.stringify(benchmark.payload, null, 2)}</pre>
            <button
              disabled={loadingKey === benchmark.endpoint}
              onClick={() => void runBenchmark(benchmark)}
            >
              {loadingKey === benchmark.endpoint ? "Executando..." : "Executar"}
            </button>
          </article>
        ))}
      </div>

      <article className="data-card">
        <h2>Resultado</h2>
        <pre>{JSON.stringify(selectedResult, null, 2)}</pre>
      </article>
    </section>
  );
}
