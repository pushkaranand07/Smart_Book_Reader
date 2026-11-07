import { useState } from "react";
import { ApiError } from "../api/errors";
import type { BookSummary } from "../api/types";
import { BookList } from "../components/BookList";
import { BookWorkspace } from "../components/BookWorkspace";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";
import { ProcessingStatus } from "../components/ProcessingStatus";
import { UploadPanel } from "../components/UploadPanel";
import { useBook } from "../hooks/useBooks";
import { isTerminal, jobErrorMessage, useJob } from "../hooks/useJob";
import { useUploadBook } from "../hooks/useUploadBook";

interface ReaderPageProps {
  /** Job polling interval; shortened in tests. */
  pollIntervalMs?: number;
}

export function ReaderPage({ pollIntervalMs }: ReaderPageProps) {
  const upload = useUploadBook();
  const [jobId, setJobId] = useState<string | null>(null);
  const [openedBookId, setOpenedBookId] = useState<string | null>(null);
  const [uploadName, setUploadName] = useState<string | null>(null);
  const job = useJob(jobId, pollIntervalMs);

  const jobData = job.data;
  const processedBookId = jobData?.status === "succeeded" ? jobData.book_id : null;
  const bookId = openedBookId ?? processedBookId;
  const book = useBook(bookId);

  const uploading = upload.isPending;
  const processing = jobId !== null && !job.isError && !isTerminal(jobData);
  const busy = uploading || processing;

  const startUpload = (file: File) => {
    setOpenedBookId(null);
    setJobId(null);
    setUploadName(file.name);
    upload.mutate(file, {
      onSuccess: (accepted) => {
        setJobId(accepted.job_id);
      },
    });
  };

  const openBook = (summary: BookSummary) => {
    upload.reset();
    setJobId(null);
    setOpenedBookId(summary.book_id);
  };

  const announcement = uploading
    ? `Uploading ${uploadName ?? "PDF"}: ${Math.floor(upload.progress * 10) * 10}%`
    : processing
      ? (jobData?.message ?? "Processing document")
      : jobData?.status === "failed"
        ? `Processing failed: ${jobErrorMessage(jobData)}`
        : book.data
          ? `${book.data.filename} is ready`
          : "";

  return (
    <>
      <main className="workspace" id="main" tabIndex={-1}>
        <section className="intro-row" aria-labelledby="intro-heading">
          <div>
            <p className="eyebrow">Document workspace</p>
            <h1 id="intro-heading">
              Read the page,
              <br />
              <em>see the meaning.</em>
            </h1>
            <p className="lede">
              Upload a textbook or paper to search its words, figures, and visual structure in one calm workspace.
            </p>
          </div>
          <UploadPanel busy={busy} onFile={startUpload} />
        </section>

        <div className="visually-hidden" aria-live="polite" aria-atomic="true" data-testid="status-announcer">
          {announcement}
        </div>

        {upload.isError && (
          <ErrorBanner
            error={upload.error}
            title="Upload failed"
            onDismiss={() => {
              upload.reset();
            }}
          />
        )}
        {job.isError && <ErrorBanner error={job.error} title="Could not check processing status" />}
        {jobData?.status === "failed" && (
          <ErrorBanner
            error={new ApiError(0, "JOB_FAILED", jobErrorMessage(jobData))}
            title="Processing failed"
            onDismiss={() => {
              setJobId(null);
            }}
          />
        )}

        {busy && (
          <ProcessingStatus
            phase={uploading ? "uploading" : "processing"}
            uploadProgress={upload.progress}
            job={jobData}
            filename={uploadName}
          />
        )}

        {!busy && bookId === null && (
          <EmptyState title="Your document starts here">
            Choose a PDF to activate page analysis, grounded search, and figure discovery.
          </EmptyState>
        )}

        {bookId !== null && book.isPending && (
          <section className="loading-state" aria-label="Loading book">
            <div className="loader" aria-hidden="true" />
            <div className="loading-state__body">
              <h2>Opening your book</h2>
              <p>Loading pages and figures...</p>
            </div>
          </section>
        )}
        {book.isError && (
          <ErrorBanner
            error={book.error}
            title="Could not open this book"
            onRetry={() => {
              void book.refetch();
            }}
          />
        )}

        {book.data && <BookWorkspace key={book.data.book_id} book={book.data} />}

        <BookList activeBookId={bookId} onOpen={openBook} />
      </main>
      <footer className="footer">
        <span>SMART READER / {book.data?.filename ?? "NO DOCUMENT LOADED"}</span>
        <span>Pipeline ready for exploration</span>
      </footer>
    </>
  );
}
