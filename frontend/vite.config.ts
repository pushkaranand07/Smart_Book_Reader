import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv, type ProxyOptions } from "vite";

const DEFAULT_API_PORT = "8000";

export default defineConfig(({ mode }) => {
  // Load every variable (not only VITE_*) so process env like VITE_API_PORT set by a shell or CI is honoured.
  const env = { ...loadEnv(mode, process.cwd(), ""), ...process.env };
  const target = env.VITE_API_PROXY_TARGET || `http://127.0.0.1:${env.VITE_API_PORT || DEFAULT_API_PORT}`;
  const backend: ProxyOptions = { target, changeOrigin: true };
  const proxy = { "/api": backend, "/healthz": backend, "/readyz": backend };

  return {
    plugins: [react()],
    server: { host: "127.0.0.1", port: 5173, proxy },
    preview: { host: "127.0.0.1", port: 4173, proxy },
    build: {
      target: "es2022",
      sourcemap: false,
    },
  };
});
