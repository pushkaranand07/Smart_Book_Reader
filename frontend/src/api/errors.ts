import type { ApiErrorBody } from "./types";

const STATUS_FALLBACKS: Record<number, { code: string; message: string }> = {
  400: { code: "BAD_REQUEST", message: "The request was not valid." },
  401: { code: "UNAUTHORIZED", message: "An API key is required. Open settings and enter your key." },
  403: { code: "FORBIDDEN", message: "This API key is not allowed to do that." },
  404: { code: "NOT_FOUND", message: "That item could not be found." },
  413: { code: "FILE_TOO_LARGE", message: "The file is too large to upload." },
  415: { code: "UNSUPPORTED_MEDIA_TYPE", message: "Only PDF files can be uploaded." },
  422: { code: "UNPROCESSABLE", message: "The document could not be accepted (for example, too many pages)." },
  429: { code: "RATE_LIMITED", message: "Too many requests. Please wait a moment and try again." },
};

/** Typed error for every failed API call. Mirrors `{error: {code, message, request_id}}`. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;

  constructor(status: number, code: string, message: string, requestId: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }

  get isUnauthorized(): boolean {
    return this.status === 401;
  }
}

export function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== "object" || value === null || !("error" in value)) return false;
  const error = value.error;
  return (
    typeof error === "object" &&
    error !== null &&
    typeof (error as Record<string, unknown>).code === "string" &&
    typeof (error as Record<string, unknown>).message === "string"
  );
}

/** Build an ApiError from an HTTP status and a (possibly non-JSON) parsed body. */
export function toApiError(status: number, body: unknown, headerRequestId: string | null = null): ApiError {
  if (isApiErrorBody(body)) {
    const { code, message, request_id } = body.error;
    return new ApiError(status, code, message, typeof request_id === "string" ? request_id : headerRequestId);
  }
  const fallback = STATUS_FALLBACKS[status] ?? {
    code: status >= 500 ? "SERVER_ERROR" : "HTTP_ERROR",
    message: status >= 500 ? "The server ran into a problem. Please try again." : `Request failed (${status}).`,
  };
  return new ApiError(status, fallback.code, fallback.message, headerRequestId);
}

export function networkError(): ApiError {
  return new ApiError(0, "NETWORK_ERROR", "Could not reach the server. Check your connection and the API URL.");
}

/** Normalise anything thrown into display-ready fields. */
export function describeError(error: unknown): { code: string; message: string; requestId: string | null } {
  if (error instanceof ApiError) return { code: error.code, message: error.message, requestId: error.requestId };
  if (error instanceof Error) return { code: "CLIENT_ERROR", message: error.message, requestId: null };
  return { code: "UNKNOWN_ERROR", message: "Something went wrong.", requestId: null };
}
