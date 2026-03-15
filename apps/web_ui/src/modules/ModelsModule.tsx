import { useEffect, useState } from "react";

import { apiRequest } from "../lib/httpClient";
import type { ApiJson } from "../lib/types";

export function ModelsModule() {
  const [models, setModels] = useState<ApiJson | null>(null);
  const [datasets, setDatasets] = useState<ApiJson | null>(null);
  const [evaluation, setEvaluation] = useState<ApiJson | null>(null);
  const [modelId, setModelId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function loadCatalogs() {
    setError(null);
    try {
      const [modelsPayload, datasetsPayload] = await Promise.all([
        apiRequest<ApiJson>("/v1/models"),
        apiRequest<ApiJson>("/v1/datasets"),
      ]);
      setModels(modelsPayload);
      setDatasets(datasetsPayload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao carregar modelos e datasets.");
    }
  }

  useEffect(() => {
    void loadCatalogs();
  }, []);

  async function trainPolicyTable() {
    setLoading(true);
    setError(null);
    try {
      await apiRequest<ApiJson>("/v1/models/train/policy-table", { method: "POST" });
      await loadCatalogs();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao treinar model.");
    } finally {
      setLoading(false);
    }
  }

  async function evaluateDataset() {
    if (!modelId || !datasetId) {
      setError("Informe model_id e dataset_id para avaliar.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const payload = await apiRequest<ApiJson>(
        `/v1/models/${encodeURIComponent(modelId)}/evaluate-dataset/${encodeURIComponent(datasetId)}`,
        { method: "POST" }
      );
      setEvaluation(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha na avaliação.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="module-grid">
      <div className="hero-card">
        <div className="hero-chip">♠ Models</div>
        <h1>Modelos e Datasets</h1>
        <p>Treino tabular local e avaliação de qualidade em dataset para ciclo rápido de IA aplicada.</p>
      </div>

      {error ? <p className="error-box">{error}</p> : null}

      <article className="data-card">
        <h2>Ações</h2>
        <div className="button-row">
          <button disabled={loading} onClick={() => void trainPolicyTable()}>
            Treinar policy-table
          </button>
          <button disabled={loading} onClick={() => void loadCatalogs()}>
            Atualizar catálogos
          </button>
        </div>
        <div className="inline-form">
          <label>
            model_id
            <input value={modelId} onChange={(event) => setModelId(event.target.value)} />
          </label>
          <label>
            dataset_id
            <input value={datasetId} onChange={(event) => setDatasetId(event.target.value)} />
          </label>
          <button disabled={loading} onClick={() => void evaluateDataset()}>
            Avaliar dataset
          </button>
        </div>
      </article>

      <div className="card-grid">
        <article className="data-card">
          <h2>Models</h2>
          <pre>{JSON.stringify(models, null, 2)}</pre>
        </article>
        <article className="data-card">
          <h2>Datasets</h2>
          <pre>{JSON.stringify(datasets, null, 2)}</pre>
        </article>
        <article className="data-card">
          <h2>Evaluation</h2>
          <pre>{JSON.stringify(evaluation, null, 2)}</pre>
        </article>
      </div>
    </section>
  );
}
