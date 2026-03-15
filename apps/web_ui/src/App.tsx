import { useMemo, useState } from "react";
import { Navigate, NavLink, Route, Routes, useNavigate } from "react-router-dom";

import type { ModuleId } from "./lib/types";
import { BenchmarksModule } from "./modules/BenchmarksModule";
import { HandsModule } from "./modules/HandsModule";
import { ModelsModule } from "./modules/ModelsModule";
import { OverviewModule } from "./modules/OverviewModule";
import { SessionsModule } from "./modules/SessionsModule";

type NavItem = {
  id: ModuleId;
  path: string;
  label: string;
  icon: string;
};

const NAV_ITEMS: NavItem[] = [
  { id: "overview", path: "/overview", label: "Overview", icon: "⚡" },
  { id: "hands", path: "/hands", label: "Hands Lab", icon: "♦" },
  { id: "sessions", path: "/sessions", label: "Sessions Arena", icon: "♣" },
  { id: "benchmarks", path: "/benchmarks", label: "Benchmarks", icon: "♥" },
  { id: "models", path: "/models", label: "Models", icon: "♠" },
];

function App() {
  const navigate = useNavigate();
  const [quickHandId, setQuickHandId] = useState("");
  const [quickSessionId, setQuickSessionId] = useState("");

  const cards = useMemo(() => ["♠", "♥", "♣", "♦"], []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <h1>mr_poker UI</h1>
          <p>Tech • IA • ML • Futuro</p>
        </div>
        <nav aria-label="Navegação principal">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.id}
              to={item.path}
              className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
            >
              <span>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <main className="content">
        <header className="topbar">
          <div className="chips" aria-hidden="true">
            {cards.map((card, index) => (
              <span key={`${card}-${index}`} className="chip">
                {card}
              </span>
            ))}
          </div>
          <div className="quick-jump">
            <label>
              Jump hand_id
              <input
                value={quickHandId}
                onChange={(event) => setQuickHandId(event.target.value)}
                placeholder="hand_id"
              />
            </label>
            <button
              onClick={() => {
                if (!quickHandId) return;
                navigate(`/hands?handId=${encodeURIComponent(quickHandId)}`);
              }}
            >
              Ir para Hands
            </button>
            <label>
              Jump session_id
              <input
                value={quickSessionId}
                onChange={(event) => setQuickSessionId(event.target.value)}
                placeholder="session_id"
              />
            </label>
            <button
              onClick={() => {
                if (!quickSessionId) return;
                navigate(`/sessions?sessionId=${encodeURIComponent(quickSessionId)}`);
              }}
            >
              Ir para Sessions
            </button>
          </div>
        </header>

        <div className="module-container">
          <Routes>
            <Route path="/" element={<Navigate to="/overview" replace />} />
            <Route path="/overview" element={<OverviewModule />} />
            <Route path="/hands" element={<HandsModule />} />
            <Route path="/sessions" element={<SessionsModule />} />
            <Route path="/benchmarks" element={<BenchmarksModule />} />
            <Route path="/models" element={<ModelsModule />} />
            <Route path="*" element={<Navigate to="/overview" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

export default App;
