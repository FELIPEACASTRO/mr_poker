import { FormEvent, useEffect, useState } from "react";

import { apiRequest } from "../lib/httpClient";
import type { ApiJson, HealthPayload } from "../lib/types";

export function OverviewModule() {
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [readiness, setReadiness] = useState<ApiJson | null>(null);
  const [alphaCandidate, setAlphaCandidate] = useState<ApiJson | null>(null);
  const [modelId, setModelId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    async function loadBootstrap() {
      setLoading(true);
      setError(null);
      try {
        const [healthPayload, readinessPayload] = await Promise.all([
          apiRequest<HealthPayload>("/health"),
          apiRequest<ApiJson>("/v1/system/readiness"),
        ]);
        if (!active) return;
        setHealth(healthPayload);
        setReadiness(readinessPayload);
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Falha ao carregar overview.");
      } finally {
        if (active) setLoading(false);
      }
    }
    void loadBootstrap();
    return () => {
      active = false;
    };
  }, []);

  async function onLoadAlphaCandidate(event: FormEvent) {
    event.preventDefault();
    if (!modelId || !datasetId) {
      setError("Informe model_id e dataset_id para consultar alpha-candidate.");
      return;
    }
    setError(null);
    try {
      const payload = await apiRequest<ApiJson>(
        `/v1/system/alpha-candidate?model_id=${encodeURIComponent(modelId)}&dataset_id=${encodeURIComponent(datasetId)}`
      );
      setAlphaCandidate(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao consultar alpha-candidate.");
    }
  }

  return (
    <section className="module-grid">
      <div className="hero-card">
        <div className="hero-chip">♠ IA • ML • Futuro • Inovação</div>
        <h1>Overview Operacional</h1>
        <p>
          Painel rápido para validar saúde do ambiente local, readiness atual e status de candidate
          promotion.
        </p>
      </div>

      {loading ? <p className="hint">Carregando dados iniciais...</p> : null}
      {error ? <p className="error-box">{error}</p> : null}

      <div className="card-grid">
        <article className="data-card">
          <h2>Health</h2>
          <pre>{JSON.stringify(health, null, 2)}</pre>
        </article>
        <article className="data-card">
          <h2>Readiness</h2>
          <pre>{JSON.stringify(readiness, null, 2)}</pre>
        </article>
      </div>

      <article className="data-card">
        <h2>Alpha Candidate</h2>
        <form className="inline-form" onSubmit={onLoadAlphaCandidate}>
          <label>
            model_id
            <input value={modelId} onChange={(event) => setModelId(event.target.value)} />
          </label>
          <label>
            dataset_id
            <input value={datasetId} onChange={(event) => setDatasetId(event.target.value)} />
          </label>
          <button type="submit">Consultar</button>
        </form>
        <pre>{JSON.stringify(alphaCandidate, null, 2)}</pre>
      </article>
    </section>
  );
}
