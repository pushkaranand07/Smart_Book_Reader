import type { Page, Route } from "@playwright/test";
import type { Answer, BookDetail, BookSummary, Figure, Job } from "../src/api/types";

/** Deterministic API fixtures that follow PLAN.md "API contract (v2)". */

export const MOCK_API_KEY = "e2e-key";
const BOOK_ID = "book-e2e";
const JOB_ID = "job-e2e";

// 1x1 transparent PNG.
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);

const figure: Figure = {
  figure_key: "p2_fig1",
  figure_id: "fig-2-1",
  label: "Figure 2.1",
  caption: "Light-dependent reactions in the thylakoid membrane",
  page_number: 2,
  image_url: `/api/books/${BOOK_ID}/figures/p2_fig1/image?exp=4102444800&sig=e2e`,
};

const uncaptioned: Figure = {
  figure_key: "p3_fig1",
  figure_id: "fig-3-1",
  label: "Figure 3.1",
  caption: null,
  page_number: 3,
  image_url: `/api/books/${BOOK_ID}/figures/p3_fig1/image?exp=4102444800&sig=e2e`,
};

const summary: BookSummary = {
  book_id: BOOK_ID,
  filename: "photosynthesis.pdf",
  status: "succeeded",
  total_pages: 3,
  created_at: "2026-11-07T09:30:00Z",
};

const detail: BookDetail = {
  ...summary,
  digital_pages: 2,
  scanned_pages: 1,
  pages: [
    {
      page_number: 1,
      page_type: "Digital",
      status: "ok",
      text: "Chapter 1\nPhotosynthesis converts light energy into chemical energy.",
      word_count: 9,
      errors: [],
      figures: [],
    },
    {
      page_number: 2,
      page_type: "Digital",
      status: "ok",
      text: "The light-dependent reactions take place in the thylakoid membranes and produce ATP and NADPH.",
      word_count: 15,
      errors: [],
      figures: [figure],
    },
    {
      page_number: 3,
      page_type: "Scanned",
      status: "degraded",
      text: "The Calvin cycle fixes carbon dioxide in the stroma.",
      word_count: 9,
      errors: [{ code: "LOW_OCR_CONFIDENCE", message: "OCR confidence below threshold" }],
      figures: [uncaptioned],
    },
  ],
};

const answer: Answer = {
  answer: "The light-dependent reactions occur in the **thylakoid membranes**, producing ATP and NADPH: $E = h\\nu$.",
  extractive_answer: "in the thylakoid membranes and produce ATP and NADPH",
  is_sufficient: true,
  confidence: 0.87,
  mode: "offline",
  citations: [
    {
      page_number: 2,
      section: "1.2 Light-dependent reactions",
      chunk_id: "chunk-2-0",
      snippet: "The light-dependent reactions take place in the thylakoid membranes and produce ATP and NADPH.",
      figure_ids: ["fig-2-1"],
    },
  ],
  figures: [figure],
};

const jobSteps: Job[] = [
  { job_id: JOB_ID, book_id: BOOK_ID, status: "queued", progress: 0, message: "Queued", error: null },
  {
    job_id: JOB_ID,
    book_id: BOOK_ID,
    status: "running",
    progress: 0.4,
    message: "Extracting page 2 of 3",
    error: null,
  },
  { job_id: JOB_ID, book_id: BOOK_ID, status: "succeeded", progress: 1, message: "Done", error: null },
];

function json(route: Route, status: number, body: unknown) {
  return route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

function apiError(route: Route, status: number, code: string, message: string) {
  return json(route, status, { error: { code, message, request_id: `req-${code.toLowerCase()}` } });
}

/** Install page.route handlers. Returns counters so tests can assert on the traffic. */
export async function mockApi(page: Page) {
  const state = { uploaded: false, jobPolls: 0, questions: [] as string[] };

  await page.route("**/healthz", (route) => json(route, 200, { status: "ok" }));
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();

    if (path.endsWith("/image")) {
      return route.fulfill({ status: 200, contentType: "image/png", body: PNG });
    }
    if ((await request.headerValue("x-api-key")) !== MOCK_API_KEY) {
      return apiError(route, 401, "UNAUTHORIZED", "Missing or invalid API key");
    }
    if (path === "/api/books" && method === "GET") {
      return json(route, 200, { books: state.uploaded ? [summary] : [] });
    }
    if (path === "/api/books" && method === "POST") {
      state.uploaded = true;
      return json(route, 202, { book_id: BOOK_ID, job_id: JOB_ID, status: "queued" });
    }
    if (path === `/api/jobs/${JOB_ID}`) {
      const step = jobSteps[Math.min(state.jobPolls, jobSteps.length - 1)];
      state.jobPolls += 1;
      return json(route, 200, step);
    }
    if (path === `/api/books/${BOOK_ID}` && method === "GET") {
      return json(route, 200, detail);
    }
    if (path === `/api/books/${BOOK_ID}/questions` && method === "POST") {
      const body = request.postDataJSON() as { question: string };
      state.questions.push(body.question);
      return json(route, 200, answer);
    }
    return apiError(route, 404, "NOT_FOUND", `No mock for ${method} ${path}`);
  });

  return state;
}
