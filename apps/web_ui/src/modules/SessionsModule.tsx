import { FormEvent, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { apiRequest } from "../lib/httpClient";
import type { ApiJson } from "../lib/types";

export function SessionsModule() {
  const [searchParams] = useSearchParams();
  const [sessionId, setSessionId] = useState(searchParams.get("sessionId") ?? "");
  const [sessionPayload, setSessionPayload] = useState<ApiJson | null>(null);
  const [analytics, setAnalytics] = useState<ApiJson | null>(null);
  const [traces, setTraces] = useState<ApiJson | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const requestedSessionId = searchParams.get("sessionId");
    if (requestedSessionId && requestedSessionId !== sessionId) {
      setSessionId(requestedSessionId);
      void loadSessionPack(requestedSessionId);
    }
  }, [searchParams, sessionId]);

  async function onRunSession(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload = await apiRequest<ApiJson>("/v1/sessions/h2h", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          num_hands: 20,
          seed_base: Date.now() % 100000,
          session_name: `ui_session_${Date.now() % 10000}`,
        }),
      });
      const nextSessionId = String(payload.session_id ?? "");
      setSessionId(nextSessionId);
      setSessionPayload(payload);
      await loadSessionPack(nextSessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao iniciar sessão.");
    } finally {
      setLoading(false);
    }
  }

  async function loadSessionPack(targetSessionId: string = sessionId) {
    if (!targetSessionId) return;
    setError(null);
    try {
      const [sessionData, analyticsData, tracesData] = await Promise.all([
        apiRequest<ApiJson>(`/v1/sessions/${encodeURIComponent(targetSessionId)}`),
        apiRequest<ApiJson>(`/v1/sessions/${encodeURIComponent(targetSessionId)}/analytics`),
        apiRequest<ApiJson>(`/v1/sessions/${encodeURIComponent(targetSessionId)}/traces`),
      ]);
      setSessionPayload(sessionData);
      setAnalytics(analyticsData);
      setTraces(tracesData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao carregar sessão.");
    }
  }

  return (
    <section className="module-grid">
      <div className="hero-card">
        <div className="hero-chip">♣ Sessions Arena</div>
        <h1>Arena de Sessões</h1>
        <p>Execute sessões H2H, acompanhe analytics e audite traces do ciclo de decisão.</p>
      </div>

      {error ? <p className="error-box">{error}</p> : null}
      {loading ? <p className="hint">Executando sessão...</p> : null}

      <article className="data-card">
        <h2>Controles</h2>
        <form onSubmit={onRunSession}>
          <button type="submit">Rodar H2H</button>
        </form>
        <div className="button-row">
          <button disabled={!sessionId} onClick={() => void loadSessionPack()}>
            Atualizar sessão
          </button>
        </div>
        <label>
          session_id ativo
          <input
            value={sessionId}
            onChange={(event) => setSessionId(event.target.value)}
            placeholder="session_id"
          />
        </label>
      </article>

      <div className="card-grid">
        <article className="data-card">
          <h2>Session</h2>
          <pre>{JSON.stringify(sessionPayload, null, 2)}</pre>
        </article>
        <article className="data-card">
          <h2>Analytics</h2>
          <pre>{JSON.stringify(analytics, null, 2)}</pre>
        </article>
        <article className="data-card">
          <h2>Traces</h2>
          <pre>{JSON.stringify(traces, null, 2)}</pre>
        </article>
      </div>
    </section>
  );
}
