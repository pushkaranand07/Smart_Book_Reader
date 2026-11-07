import { QueryClientProvider, type QueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { TopBar } from "./components/TopBar";
import { useHealth } from "./hooks/useHealth";
import { createQueryClient } from "./lib/queryClient";
import { ReaderPage } from "./pages/ReaderPage";

function Shell({ pollIntervalMs }: { pollIntervalMs?: number | undefined }) {
  const health = useHealth();
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <TopBar health={health} />
      <ReaderPage {...(pollIntervalMs === undefined ? {} : { pollIntervalMs })} />
    </div>
  );
}

interface AppProps {
  queryClient?: QueryClient;
  pollIntervalMs?: number;
}

export function App({ queryClient, pollIntervalMs }: AppProps) {
  const [client] = useState(() => queryClient ?? createQueryClient());
  return (
    <QueryClientProvider client={client}>
      <Shell pollIntervalMs={pollIntervalMs} />
    </QueryClientProvider>
  );
}
