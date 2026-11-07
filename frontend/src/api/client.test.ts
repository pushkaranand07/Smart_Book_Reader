import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BOOK_ID, answer, bookDetail, bookSummary, errorBody, job, jsonResponse } from "../test/fixtures";
import { MockXhr } from "../test/mockXhr";
import { apiKeyStore, API_KEY_STORAGE_KEY } from "./apiKeyStore";
import { askQuestion, checkHealth, getBook, getJob, listBooks, uploadBook } from "./client";
import { apiUrl, resolveApiBaseUrl } from "./config";
import { ApiError } from "./errors";

function mockFetch(...responses: Response[]) {
  const fn = vi.fn<typeof fetch>();
  for (const response of responses) fn.mockResolvedValueOnce(response);
  vi.stubGlobal("fetch", fn);
  return fn;
}

function headersOf(fn: ReturnType<typeof mockFetch>, call = 0): Record<string, string> {
  return (fn.mock.calls[call]?.[1]?.headers ?? {}) as Record<string, string>;
}

describe("config", () => {
  it("defaults to same origin and strips trailing slashes", () => {
    expect(resolveApiBaseUrl(undefined)).toBe("");
    expect(resolveApiBaseUrl(" https://api.example.com/// ")).toBe("https://api.example.com");
  });

  it("prefixes signed media paths with the base URL", () => {
    const signed = "/api/books/b/figures/k/image?exp=1&sig=x";
    expect(apiUrl(signed, "https://api.example.com")).toBe(`https://api.example.com${signed}`);
    expect(apiUrl(signed, "")).toBe(signed);
    expect(apiUrl("https://cdn.example.com/a.png", "https://api.example.com")).toBe("https://cdn.example.com/a.png");
  });
});

describe("request helpers", () => {
  it("sends X-API-Key from sessionStorage on every request", async () => {
    apiKeyStore.setApiKey("secret-key");
    expect(window.sessionStorage.getItem(API_KEY_STORAGE_KEY)).toBe("secret-key");
    const fetchMock = mockFetch(jsonResponse({ books: [bookSummary] }), jsonResponse(bookDetail));

    await expect(listBooks()).resolves.toEqual({ books: [bookSummary] });
    await expect(getBook(BOOK_ID)).resolves.toEqual(bookDetail);

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/books");
    expect(fetchMock.mock.calls[1]?.[0]).toBe(`/api/books/${BOOK_ID}`);
    expect(headersOf(fetchMock, 0)["X-API-Key"]).toBe("secret-key");
    expect(headersOf(fetchMock, 1)["X-API-Key"]).toBe("secret-key");
  });

  it("omits the header when no key is set", async () => {
    const fetchMock = mockFetch(jsonResponse(job()));
    await getJob("job-456");
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/jobs/job-456");
    expect(headersOf(fetchMock)).not.toHaveProperty("X-API-Key");
  });

  it("posts questions as JSON without any Gemini key", async () => {
    const fetchMock = mockFetch(jsonResponse(answer));
    await expect(askQuestion(BOOK_ID, "What is a cell wall?")).resolves.toEqual(answer);
    const init = fetchMock.mock.calls[0]?.[1];
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({ question: "What is a cell wall?" });
    expect(headersOf(fetchMock)["Content-Type"]).toBe("application/json");
  });

  it("parses the typed error envelope into ApiError", async () => {
    mockFetch(jsonResponse(errorBody("BOOK_NOT_FOUND", "No such book", "req-42"), 404));
    const error = await getBook("missing").catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 404, code: "BOOK_NOT_FOUND", message: "No such book", requestId: "req-42" });
  });

  it("falls back to a status-based error for non-JSON bodies", async () => {
    mockFetch(new Response("<html>Bad gateway</html>", { status: 502, headers: { "X-Request-ID": "hdr-1" } }));
    await expect(listBooks()).rejects.toMatchObject({ status: 502, code: "SERVER_ERROR", requestId: "hdr-1" });
  });

  it("maps network failures to NETWORK_ERROR", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(listBooks()).rejects.toMatchObject({ status: 0, code: "NETWORK_ERROR" });
  });

  it("flags the key as required on 401", async () => {
    mockFetch(jsonResponse(errorBody("UNAUTHORIZED", "Missing API key"), 401));
    await expect(listBooks()).rejects.toMatchObject({ status: 401 });
    expect(apiKeyStore.getSnapshot().keyRequired).toBe(true);
  });

  it("checks health at /healthz", async () => {
    const fetchMock = mockFetch(jsonResponse({ status: "ok" }));
    await expect(checkHealth()).resolves.toBe(true);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/healthz");
  });
});

describe("uploadBook (XHR)", () => {
  beforeEach(() => {
    MockXhr.reset();
    vi.stubGlobal("XMLHttpRequest", MockXhr);
  });

  afterEach(() => {
    MockXhr.reset();
  });

  it("reports upload progress, sends the key and resolves with the 202 body", async () => {
    apiKeyStore.setApiKey("upload-key");
    const progress: number[] = [];
    const file = new File(["%PDF-1.7"], "biology.pdf", { type: "application/pdf" });
    const promise = uploadBook(file, { onProgress: (value) => progress.push(value) });

    const xhr = MockXhr.latest();
    expect(xhr.method).toBe("POST");
    expect(xhr.url).toBe("/api/books");
    expect(xhr.headers["X-API-Key"]).toBe("upload-key");
    expect(xhr.body).toBeInstanceOf(FormData);
    expect((xhr.body as FormData).get("file")).toBeInstanceOf(File);

    xhr.progress(25, 100);
    xhr.progress(75, 100);
    xhr.respond(202, { book_id: BOOK_ID, job_id: "job-456", status: "queued" });

    await expect(promise).resolves.toEqual({ book_id: BOOK_ID, job_id: "job-456", status: "queued" });
    expect(progress).toEqual([0.25, 0.75, 1]);
  });

  it.each([
    [413, "FILE_TOO_LARGE", "File exceeds 50 MB"],
    [415, "NOT_A_PDF", "Only PDF files are accepted"],
    [422, "TOO_MANY_PAGES", "Book has more than 1500 pages"],
  ])("rejects with a typed error on %i", async (status, code, message) => {
    const promise = uploadBook(new File(["x"], "x.pdf"));
    MockXhr.latest().respond(status, errorBody(code, message, "req-up"));
    await expect(promise).rejects.toMatchObject({ status, code, message, requestId: "req-up" });
  });

  it("rejects with NETWORK_ERROR when the connection fails", async () => {
    const promise = uploadBook(new File(["x"], "x.pdf"));
    MockXhr.latest().fail();
    await expect(promise).rejects.toMatchObject({ code: "NETWORK_ERROR" });
  });

  it("aborts the XHR when the signal fires", async () => {
    const controller = new AbortController();
    const promise = uploadBook(new File(["x"], "x.pdf"), { signal: controller.signal });
    controller.abort();
    await expect(promise).rejects.toMatchObject({ name: "AbortError" });
    expect(MockXhr.latest().aborted).toBe(true);
  });
});
