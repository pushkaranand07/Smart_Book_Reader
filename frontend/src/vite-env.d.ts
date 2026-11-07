/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** API origin; empty = same origin. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
