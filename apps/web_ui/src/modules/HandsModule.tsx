import { FormEvent, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { apiRequest } from "../lib/httpClient";
import type { ApiJson } from "../lib/types";

const ACTION_OPTIONS = ["fold", "check", "call", "bet", "raise", "all_in"];

export function HandsModule() {
  const [searchParams] = useSearchParams();
  const [handId, setHandId] = useState(searchParams.get("handId") ?? "");
  const [snapshot, setSnapshot] = useState<ApiJson | null>(null);
  const [review, setReview] = useState<ApiJson | null>(null);
  const [actionType, setActionType] = useState("check");
  const [actionAmount, setActionAmount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const requestedHandId = searchParams.get("handId");
    if (requestedHandId && requestedHandId !== handId) {
      setHandId(requestedHandId);
      void fetchSnapshot(requestedHandId);
    }
  }, [searchParams, handId]);

  async function fetchSnapshot(targetHandId: string = handId) {
    if (!targetHandId) return;
    setLoading(true);
    setError(null);
    try {
      const payload = await apiRequest<ApiJson>(`/v1/hands/${encodeURIComponent(targetHandId)}`);
      setSnapshot(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao carregar mão.");
    } finally {
      setLoading(false);
    }
  }

  async function onCreateHand(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const payload = await apiRequest<ApiJson>("/v1/hands/new", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stacks: [100, 100],
          button_seat: 0,
          seed: Date.now() % 100000,
        }),
      });
      const nextHandId = String(payload.hand_id ?? "");
      setHandId(nextHandId);
      setSnapshot(payload);
      setReview(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao criar mão.");
    } finally {
      setLoading(false);
    }
  }

  async function onAutoAction() {
    if (!handId) return;
    setError(null);
    try {
      const payload = await apiRequest<ApiJson>(`/v1/hands/${encodeURIComponent(handId)}/auto`, {
        method: "POST",
      });
      setSnapshot(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha no auto action.");
    }
  }

  async function onManualAction(event: FormEvent) {
    event.preventDefault();
    if (!handId) return;
    setError(null);
    try {
      const payload = await apiRequest<ApiJson>(`/v1/hands/${encodeURIComponent(handId)}/actions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action_type: actionType,
          amount: Number(actionAmount),
        }),
      });
      setSnapshot(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha no action manual.");
    }
  }

  async function onLoadReview() {
    if (!handId) return;
    setError(null);
    try {
      const payload = await apiRequest<ApiJson>(`/v1/hands/${encodeURIComponent(handId)}/review`);
      setReview(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao carregar review.");
    }
  }

  return (
    <section className="module-grid">
      <div className="hero-card">
        <div className="hero-chip">♦ Hands Lab</div>
        <h1>Laboratório de Mãos</h1>
        <p>Crie hands, execute ações automáticas/manuais e revise rapidamente decisões.</p>
      </div>

      {error ? <p className="error-box">{error}</p> : null}
      {loading ? <p className="hint">Processando...</p> : null}

      <article className="data-card">
        <h2>Fluxo rápido</h2>
        <form onSubmit={onCreateHand}>
          <button type="submit">Nova Hand</button>
        </form>
        <div className="button-row">
          <button disabled={!handId} onClick={() => void fetchSnapshot()}>
            Atualizar Snapshot
          </button>
          <button disabled={!handId} onClick={() => void onAutoAction()}>
            Auto Action
          </button>
          <button disabled={!handId} onClick={() => void onLoadReview()}>
            Review
          </button>
        </div>
        <label>
          hand_id ativo
          <input value={handId} onChange={(event) => setHandId(event.target.value)} placeholder="hand_id" />
        </label>
      </article>

      <article className="data-card">
        <h2>Ação Manual</h2>
        <form className="inline-form" onSubmit={onManualAction}>
          <label>
            action_type
            <select value={actionType} onChange={(event) => setActionType(event.target.value)}>
              {ACTION_OPTIONS.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label>
            amount
            <input
              type="number"
              min={0}
              value={actionAmount}
              onChange={(event) => setActionAmount(Number(event.target.value))}
            />
          </label>
          <button type="submit" disabled={!handId}>
            Enviar ação
          </button>
        </form>
      </article>

      <div className="card-grid">
        <article className="data-card">
          <h2>Snapshot</h2>
          <pre>{JSON.stringify(snapshot, null, 2)}</pre>
        </article>
        <article className="data-card">
          <h2>Review</h2>
          <pre>{JSON.stringify(review, null, 2)}</pre>
        </article>
      </div>
    </section>
  );
}
