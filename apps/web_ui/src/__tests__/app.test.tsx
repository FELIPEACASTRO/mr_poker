import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "../App";

describe("App", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renderiza navegação principal e módulo Overview", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/health")) {
          return new Response(JSON.stringify({ status: "ok", version: "test", db_path: "tmp.db" }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          });
        }
        return new Response(JSON.stringify({ production_candidate: "alpha" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      })
    );

    render(
      <MemoryRouter initialEntries={["/overview"]}>
        <App />
      </MemoryRouter>
    );

    expect(screen.getByText("mr_poker UI")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Overview/i })).toBeInTheDocument();
    expect(await screen.findByText("Overview Operacional")).toBeInTheDocument();
  });
});
