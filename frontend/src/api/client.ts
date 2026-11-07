import { apiKeyStore } from "./apiKeyStore";
import { apiUrl } from "./config";
import { ApiError, networkError, toApiError } from "./errors";
import type { Answer, BookDetail, BookList, Job, UploadAccepted } from "./types";

const API_KEY_HEADER = "X-API-Key";
const REQUEST_ID_HEADER = "X-Request-ID";

function authHeaders(): Record<string, string> {
  const key = apiKeyStore.getApiKey();
  return key ? { [API_KEY_HEADER]: key } : {};
}

function parseJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

function fail(status: number, body: unknown, requestId: string | null): never {
  const error = toApiError(status, body, requestId);
  if (error.isUnauthorized) apiKeyStore.requireKey();
  throw error;
}

export interface RequestOptions {
  method?: "GET" | "POST" | "DELETE";
  body?: unknown;
  signal?: AbortSignal | undefined;
}

/** JSON request against the API with the API key attached. Throws ApiError on any failure. */
export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json", ...authHeaders() };
  const init: RequestInit = { method: options.method ?? "GET", headers };
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  if (options.signal) init.signal = options.signal;

  let response: Response;
  try {
    response = await fetch(apiUrl(path), init);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw networkError();
  }
  const body = parseJson(await response.text());
  if (!response.ok) fail(response.status, body, response.headers.get(REQUEST_ID_HEADER));
  return body as T;
}

export function listBooks(signal?: AbortSignal): Promise<BookList> {
  return request<BookList>("/api/books", { signal });
}

export function getBook(bookId: string, signal?: AbortSignal): Promise<BookDetail> {
  return request<BookDetail>(`/api/books/${encodeURIComponent(bookId)}`, { signal });
}

export function getJob(jobId: string, signal?: AbortSignal): Promise<Job> {
  return request<Job>(`/api/jobs/${encodeURIComponent(jobId)}`, { signal });
}

export function askQuestion(bookId: string, question: string, signal?: AbortSignal): Promise<Answer> {
  return request<Answer>(`/api/books/${encodeURIComponent(bookId)}/questions`, {
    method: "POST",
    body: { question },
    signal,
  });
}

/** Liveness probe; outside /api so it never needs a key. Resolves true when the API answers 2xx. */
export async function checkHealth(signal?: AbortSignal): Promise<boolean> {
  await request<unknown>("/healthz", { signal });
  return true;
}

export interface UploadOptions {
  /** Called with 0..1 as the request body is sent. */
  onProgress?: (fraction: number) => void;
  signal?: AbortSignal | undefined;
}

/**
 * Upload a PDF with XMLHttpRequest so real upload progress events are available (fetch has none).
 * Resolves with the 202 body `{book_id, job_id, status}`.
 */
export function uploadBook(file: File, options: UploadOptions = {}): Promise<UploadAccepted> {
  const { onProgress, signal } = options;
  return new Promise<UploadAccepted>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Upload aborted", "AbortError"));
      return;
    }
    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl("/api/books"));
    xhr.setRequestHeader("Accept", "application/json");
    for (const [name, value] of Object.entries(authHeaders())) xhr.setRequestHeader(name, value);

    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable && event.total > 0) onProgress?.(Math.min(1, event.loaded / event.total));
    });
    xhr.addEventListener("load", () => {
      const body = parseJson(xhr.responseText);
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress?.(1);
        resolve(body as UploadAccepted);
        return;
      }
      try {
        fail(xhr.status, body, xhr.getResponseHeader(REQUEST_ID_HEADER));
      } catch (error) {
        reject(error instanceof ApiError ? error : networkError());
      }
    });
    xhr.addEventListener("error", () => {
      reject(networkError());
    });
    xhr.addEventListener("abort", () => {
      reject(new DOMException("Upload aborted", "AbortError"));
    });
    signal?.addEventListener("abort", () => {
      xhr.abort();
    });

    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}
