export type ApiRequestOptions = RequestInit & {
  timeoutMs?: number;
  retries?: number;
};

function shouldRetry(method: string, retries?: number): number {
  if (typeof retries === "number") {
    return retries;
  }
  return method === "GET" ? 1 : 0;
}

function buildUrl(path: string): string {
  const base = import.meta.env.VITE_API_BASE_URL ?? "";
  return `${base}${path}`;
}

function parseErrorBody(body: unknown): string {
  if (typeof body === "string") {
    return body;
  }
  if (body && typeof body === "object" && "detail" in body) {
    return String((body as { detail: unknown }).detail);
  }
  return "Erro inesperado na chamada de API.";
}

async function fetchWithTimeout(input: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, {
      ...init,
      signal: controller.signal,
    });
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export async function apiRequest<T>(path: string, options: ApiRequestOptions = {}): Promise<T> {
  const method = (options.method ?? "GET").toUpperCase();
  const retries = shouldRetry(method, options.retries);
  const timeoutMs = options.timeoutMs ?? 6000;

  let attempts = 0;
  let lastError: Error | null = null;
  while (attempts <= retries) {
    attempts += 1;
    try {
      const response = await fetchWithTimeout(buildUrl(path), options, timeoutMs);
      const contentType = response.headers.get("content-type") ?? "";
      const parsed = contentType.includes("application/json")
        ? await response.json()
        : await response.text();

      if (!response.ok) {
        throw new Error(parseErrorBody(parsed));
      }
      return parsed as T;
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        lastError = new Error("Tempo esgotado na requisição.");
      } else if (error instanceof Error) {
        lastError = error;
      } else {
        lastError = new Error("Falha desconhecida de rede.");
      }

      if (attempts > retries) {
        throw lastError;
      }
    }
  }
  throw lastError ?? new Error("Falha de rede.");
}
