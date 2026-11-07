import type { HealthStatus } from "../hooks/useHealth";
import { ApiKeySettings } from "./ApiKeySettings";

export function TopBar({ health }: { health: HealthStatus }) {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">
          SR
        </div>
        <div>
          <strong>Smart Reader</strong>
          <span>Visual document intelligence</span>
        </div>
      </div>
      <div className="topbar__right">
        <p className="topbar-status">
          <span className={`status-dot status-${health}`} aria-hidden="true" />
          API {health}
        </p>
        <ApiKeySettings />
      </div>
    </header>
  );
}
