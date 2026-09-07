const configuredApi = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");
export const API_BASE_URL = configuredApi.replace(/\/+$/, "");

export function assetUrl(path) {
  if (!path || path.startsWith("http")) return path;
  return path.startsWith("/") ? `${API_BASE_URL}${path}` : path;
}

export async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || "The server could not complete that request.");
  }
  return body;
}

export function checkHealth() {
  return request("/api/health");
}
