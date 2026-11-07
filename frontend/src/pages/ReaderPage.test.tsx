import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { BOOK_ID, JOB_ID, answer, bookDetail, bookSummary, errorBody, job, jsonResponse } from "../test/fixtures";
import { MockXhr } from "../test/mockXhr";
import { createTestQueryClient } from "../test/utils";

type Route = (url: string, init: RequestInit | undefined) => Response | undefined;

function routeFetch(route: Route) {
  const fn = vi.fn<typeof fetch>((input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (url === "/healthz") return Promise.resolve(jsonResponse({ status: "ok" }));
    const response = route(url, init);
    return Promise.resolve(response ?? jsonResponse(errorBody("NOT_FOUND", `No route for ${url}`), 404));
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

function renderApp() {
  return render(<App queryClient={createTestQueryClient()} pollIntervalMs={10} />);
}

describe("ReaderPage", () => {
  beforeEach(() => {
    MockXhr.reset();
    vi.stubGlobal("XMLHttpRequest", MockXhr);
  });

  it("shows the empty state and an empty book list", async () => {
    routeFetch((url) => (url === "/api/books" ? jsonResponse({ books: [] }) : undefined));
    renderApp();
    expect(screen.getByRole("heading", { name: "Your document starts here" })).toBeInTheDocument();
    expect(await screen.findByText("No books yet. Upload a PDF to get started.")).toBeInTheDocument();
    expect(await screen.findByText(/API connected/)).toBeInTheDocument();
  });

  it("reopens a book, answers a question and jumps to the cited page", async () => {
    const user = userEvent.setup();
    const fetchMock = routeFetch((url, init) => {
      if (url === "/api/books") return jsonResponse({ books: [bookSummary] });
      if (url === `/api/books/${BOOK_ID}`) return jsonResponse(bookDetail);
      if (url === `/api/books/${BOOK_ID}/questions` && init?.method === "POST") return jsonResponse(answer);
      return undefined;
    });
    renderApp();

    await user.click(await screen.findByRole("button", { name: /biology\.pdf/ }));
    const rail = await screen.findByRole("listbox", { name: "Pages" });
    expect(within(rail).getByRole("option", { name: /Page 1/ })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Needs review")).toBeInTheDocument();

    await user.type(screen.getByLabelText("Your question about this book"), "What do plant cells have?");
    await user.click(screen.getByRole("button", { name: /Ask/ }));

    const answerRegion = await screen.findByTestId("answer");
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/books/${BOOK_ID}/questions`,
      expect.objectContaining({ method: "POST", body: JSON.stringify({ question: "What do plant cells have?" }) }),
    );

    await user.click(within(answerRegion).getByRole("button", { name: /Page 3/ }));
    expect(within(rail).getByRole("option", { name: /Page 3/ })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByTestId("page-view")).toHaveAttribute("data-page-number", "3");
    expect(screen.getByRole("heading", { name: /Scanned page/ })).toHaveFocus();

    // Figure ids jump to the figure's page and highlight it.
    await user.click(within(answerRegion).getByRole("button", { name: "Show Figure 2.1" }));
    expect(within(rail).getByRole("option", { name: /Page 2/ })).toHaveAttribute("aria-selected", "true");
    const pageView = screen.getByTestId("page-view");
    expect(pageView.querySelector(".figure-card--highlighted")).toHaveAttribute("data-figure-id", "fig-2-1");
  });

  it("uploads with progress, polls the job and opens the processed book", async () => {
    const user = userEvent.setup();
    let polls = 0;
    routeFetch((url) => {
      if (url === "/api/books") return jsonResponse({ books: [] });
      if (url === `/api/jobs/${JOB_ID}`) {
        polls += 1;
        return jsonResponse(
          polls < 2
            ? job({ progress: 0.4, message: "Extracting page 2 of 3" })
            : job({ status: "succeeded", progress: 1 }),
        );
      }
      if (url === `/api/books/${BOOK_ID}`) return jsonResponse(bookDetail);
      return undefined;
    });
    renderApp();

    const input = screen.getByLabelText(/Choose file/);
    await user.upload(input, new File(["%PDF-1.7"], "biology.pdf", { type: "application/pdf" }));

    await waitFor(() => {
      expect(MockXhr.instances).toHaveLength(1);
    });
    act(() => {
      MockXhr.latest().progress(50, 100);
    });
    expect(await screen.findByRole("progressbar", { name: "Upload progress" })).toHaveAttribute("aria-valuenow", "50");
    expect(screen.getByTestId("status-announcer")).toHaveTextContent("Uploading biology.pdf: 50%");

    act(() => {
      MockXhr.latest().respond(202, { book_id: BOOK_ID, job_id: JOB_ID, status: "queued" });
    });

    expect(await screen.findByRole("listbox", { name: "Pages" })).toBeInTheDocument();
    expect(polls).toBeGreaterThanOrEqual(2);
    expect(screen.getByTestId("status-announcer")).toHaveTextContent("biology.pdf is ready");
  });

  it("shows a failed job as an error", async () => {
    const user = userEvent.setup();
    routeFetch((url) => {
      if (url === "/api/books") return jsonResponse({ books: [] });
      if (url === `/api/jobs/${JOB_ID}`) return jsonResponse(job({ status: "failed", error: "PDF is encrypted" }));
      return undefined;
    });
    renderApp();
    await user.upload(screen.getByLabelText(/Choose file/), new File(["x"], "locked.pdf"));
    await waitFor(() => {
      expect(MockXhr.instances).toHaveLength(1);
    });
    act(() => {
      MockXhr.latest().respond(202, { book_id: BOOK_ID, job_id: JOB_ID, status: "queued" });
    });
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Processing failed");
    expect(alert).toHaveTextContent("PDF is encrypted");
  });

  it("prompts for an API key on 401 and retries with it", async () => {
    const user = userEvent.setup();
    const fetchMock = routeFetch((url, init) => {
      const key = (init?.headers as Record<string, string> | undefined)?.["X-API-Key"];
      if (url === "/api/books") {
        return key === "good-key"
          ? jsonResponse({ books: [bookSummary] })
          : jsonResponse(errorBody("UNAUTHORIZED", "Missing or invalid API key"), 401);
      }
      return undefined;
    });
    renderApp();

    const dialog = await screen.findByRole("dialog", { name: "API access" });
    expect(within(dialog).getByRole("alert")).toHaveTextContent("401");
    const input = within(dialog).getByLabelText("API key");
    await waitFor(() => {
      expect(input).toHaveFocus();
    });
    await user.type(input, "good-key");
    await user.click(within(dialog).getByRole("button", { name: "Save key" }));

    expect(await screen.findByRole("button", { name: /biology\.pdf/ })).toBeInTheDocument();
    expect(window.sessionStorage.getItem("sbr.apiKey")).toBe("good-key");
    expect(screen.queryByRole("dialog", { name: "API access" })).not.toBeInTheDocument();
    const lastCall = fetchMock.mock.calls.filter(([url]) => url === "/api/books").at(-1);
    expect((lastCall?.[1]?.headers as Record<string, string>)["X-API-Key"]).toBe("good-key");
  });
});
