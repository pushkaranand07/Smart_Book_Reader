import { QueryClient } from "@tanstack/react-query";
import { ApiError } from "../api/errors";

const MAX_RETRIES = 2;

/** Retry transient failures only; 4xx answers (bad key, not found, too large) will not change on retry. */
export function shouldRetry(failureCount: number, error: unknown): boolean {
  if (error instanceof ApiError && error.status >= 400 && error.status < 500) return false;
  return failureCount < MAX_RETRIES;
}

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: shouldRetry, refetchOnWindowFocus: false, staleTime: 30_000 },
      mutations: { retry: false },
    },
  });
}
