import type { Answer, BookDetail, BookSummary, Figure, Job } from "../api/types";

export const BOOK_ID = "book-123";
export const JOB_ID = "job-456";

export const figure: Figure = {
  figure_key: "p2_fig1",
  figure_id: "fig-2-1",
  label: "Figure 2.1",
  caption: "Structure of a plant cell",
  page_number: 2,
  image_url: `/api/books/${BOOK_ID}/figures/p2_fig1/image?exp=1700000000&sig=abc`,
};

export const uncaptionedFigure: Figure = {
  ...figure,
  figure_key: "p3_fig1",
  figure_id: "fig-3-1",
  label: "Figure 3.1",
  caption: null,
  page_number: 3,
  image_url: `/api/books/${BOOK_ID}/figures/p3_fig1/image?exp=1700000000&sig=def`,
};

export const bookSummary: BookSummary = {
  book_id: BOOK_ID,
  filename: "biology.pdf",
  status: "succeeded",
  total_pages: 3,
  created_at: "2026-11-01T10:00:00Z",
};

export const bookDetail: BookDetail = {
  ...bookSummary,
  digital_pages: 2,
  scanned_pages: 1,
  pages: [
    {
      page_number: 1,
      page_type: "Digital",
      status: "ok",
      text: "Chapter 1 Introduction to cells.",
      word_count: 5,
      errors: [],
      figures: [],
    },
    {
      page_number: 2,
      page_type: "Digital",
      status: "ok",
      text: "Plant cells have a cell wall and chloroplasts.",
      word_count: 8,
      errors: [],
      figures: [figure],
    },
    {
      page_number: 3,
      page_type: "Scanned",
      status: "degraded",
      text: "Mitochondria produce energy.",
      word_count: 3,
      errors: [{ code: "LOW_OCR_CONFIDENCE", message: "OCR confidence below threshold" }],
      figures: [uncaptionedFigure],
    },
  ],
};

export const answer: Answer = {
  answer: "Plant cells have a **cell wall** and chloroplasts.",
  extractive_answer: "a cell wall and chloroplasts",
  is_sufficient: true,
  confidence: 0.82,
  mode: "offline",
  citations: [
    {
      page_number: 2,
      section: "1.2 Plant cells",
      chunk_id: "c-2-0",
      snippet: "Plant cells have a cell wall and chloroplasts.",
      figure_ids: ["fig-2-1"],
    },
    {
      page_number: 3,
      section: null,
      chunk_id: "c-3-0",
      snippet: "Mitochondria produce energy.",
      figure_ids: [],
    },
  ],
  figures: [figure],
};

export const insufficientAnswer: Answer = {
  answer: "The book does not discuss quantum chromodynamics.",
  extractive_answer: null,
  is_sufficient: false,
  confidence: 0.08,
  mode: "gemini",
  citations: [],
  figures: [],
};

export function job(overrides: Partial<Job> = {}): Job {
  return {
    job_id: JOB_ID,
    book_id: BOOK_ID,
    status: "running",
    progress: 0.5,
    message: "Extracting page 2 of 3",
    error: null,
    ...overrides,
  };
}

export function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

export function errorBody(code: string, message: string, requestId = "req-1") {
  return { error: { code, message, request_id: requestId } };
}
