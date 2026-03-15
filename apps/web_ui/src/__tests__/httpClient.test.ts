import { afterEach, describe, expect, it, vi } from "vitest";

import { apiRequest } from "../lib/httpClient";

describe("apiRequest", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("retenta GET uma vez quando ocorre falha transitória", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error("network"))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    vi.stubGlobal("fetch", fetchMock);

    const payload = await apiRequest<{ ok: boolean }>("/health");
    expect(payload.ok).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("não retenta POST quando ocorre falha", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error("network"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      apiRequest("/v1/hands/new", {
        method: "POST",
        body: "{}",
      })
    ).rejects.toThrow("network");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
