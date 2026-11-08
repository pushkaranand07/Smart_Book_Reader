/**
 * Types mirroring the Smart Book Reader HTTP API contract (v2).
 * Keep in sync with "API contract (v2)" in PLAN.md.
 */

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

/** Book processing status. The contract leaves the set open, so unknown strings are tolerated. */
export type BookStatus = JobStatus | (string & {});

export type PageType = "Digital" | "Scanned";
export type PageStatus = "ok" | "degraded" | "failed";
export type AnswerMode = "offline" | "gemini" | "azure_openai";

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    request_id: string;
  };
}

/** `POST /api/books` -> 202 */
export interface UploadAccepted {
  book_id: string;
  job_id: string;
  status: JobStatus;
}

/** `GET /api/jobs/{job_id}` */
export interface Job {
  job_id: string;
  book_id: string;
  status: JobStatus;
  /** 0..1 */
  progress: number;
  message: string | null;
  error: string | { code: string; message: string } | null;
}

export interface BookSummary {
  book_id: string;
  filename: string;
  status: BookStatus;
  total_pages: number;
  /** ISO-8601 timestamp */
  created_at: string;
}

export interface PageError {
  code: string;
  message: string;
}

export interface Figure {
  figure_key: string;
  figure_id: string;
  label: string;
  caption: string | null;
  page_number: number;
  /** Signed, short-lived path such as `/api/books/{id}/figures/{key}/image?exp=..&sig=..` */
  image_url: string;
}

export interface Page {
  page_number: number;
  page_type: PageType;
  status: PageStatus;
  text: string;
  word_count: number;
  errors: PageError[];
  figures: Figure[];
}

export interface BookDetail extends BookSummary {
  digital_pages: number;
  scanned_pages: number;
  pages: Page[];
}

/** `GET /api/books` */
export interface BookList {
  books: BookSummary[];
}

export interface Citation {
  page_number: number;
  section: string | null;
  chunk_id: string;
  snippet: string;
  figure_ids: string[];
}

/** `POST /api/books/{book_id}/questions` */
export interface Answer {
  answer: string;
  extractive_answer: string | null;
  is_sufficient: boolean;
  /** 0..1 */
  confidence: number;
  mode: AnswerMode;
  citations: Citation[];
  figures: Figure[];
}
