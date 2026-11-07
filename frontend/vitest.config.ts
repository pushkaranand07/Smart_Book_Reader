import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    // Integration tests drive the whole app in jsdom; leave headroom for slow or busy CI machines.
    testTimeout: 20_000,
    restoreMocks: true,
    unstubGlobals: true,
    unstubEnvs: true,
  },
});
