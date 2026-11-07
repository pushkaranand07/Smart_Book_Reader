/** API origin from `VITE_API_URL`; empty means same origin (dev proxy or nginx). */
export function resolveApiBaseUrl(raw: string | undefined): string {
  return (raw ?? "").trim().replace(/\/+$/, "");
}

export const API_BASE_URL = resolveApiBaseUrl(import.meta.env.VITE_API_URL);

/** Prefix an API path (including already-signed media paths) with the configured base URL. */
export function apiUrl(path: string, base: string = API_BASE_URL): string {
  if (/^https?:\/\//i.test(path)) return path;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${base}${normalized}`;
}
