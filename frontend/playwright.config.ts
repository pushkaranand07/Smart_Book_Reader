import { defineConfig, devices } from "@playwright/test";

/**
 * E2E runs against the production build served by `vite preview`.
 * Default: the API is mocked with page.route (deterministic fixtures, see e2e/mockApi.ts).
 * E2E_REAL_API=1: the preview server proxies /api, /healthz, /readyz to E2E_API_URL (same-origin, no CORS).
 */
const realApi = process.env.E2E_REAL_API === "1";
const port = Number(process.env.E2E_PORT ?? 4317);
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: realApi ? 15 * 60_000 : 60_000,
  expect: { timeout: realApi ? 60_000 : 10_000 },
  reporter: [["list"]],
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npm run build && npx vite preview --port ${port} --strictPort --host 127.0.0.1`,
    url: baseURL,
    // Never reuse: a stale or foreign server on the port would make results meaningless.
    reuseExistingServer: false,
    timeout: 180_000,
    env: realApi && process.env.E2E_API_URL ? { VITE_API_PROXY_TARGET: process.env.E2E_API_URL } : {},
  },
});
