import type { Job } from "../api/types";
import { ProgressBar } from "./ProgressBar";

interface ProcessingStatusProps {
  phase: "uploading" | "processing";
  uploadProgress: number;
  job: Job | undefined;
  filename: string | null;
}

const STATUS_TEXT: Record<Job["status"], string> = {
  queued: "Waiting in the processing queue",
  running: "Processing pages",
  succeeded: "Processing finished",
  failed: "Processing failed",
};

/** Upload + background-job progress. The parent wraps it in an aria-live region. */
export function ProcessingStatus({ phase, uploadProgress, job, filename }: ProcessingStatusProps) {
  const uploading = phase === "uploading";
  const value = uploading ? uploadProgress : (job?.progress ?? 0);
  const heading = uploading ? "Uploading your document" : "Preparing your reading room";
  const detail = uploading
    ? `Sending ${filename ?? "PDF"}: ${Math.round(uploadProgress * 100)}%`
    : job
      ? `${job.message ?? STATUS_TEXT[job.status]} (${Math.round(job.progress * 100)}%)`
      : "Starting processing";

  return (
    <section className="loading-state" aria-labelledby="processing-heading">
      <div className="loader" aria-hidden="true" />
      <div className="loading-state__body">
        <h2 id="processing-heading">{heading}</h2>
        <p data-testid="processing-detail">{detail}</p>
        <ProgressBar value={value} label={uploading ? "Upload progress" : "Processing progress"} />
        {!uploading && (
          <p className="loading-state__hint">
            The first run may take a moment while the vision and OCR engines inspect each page.
          </p>
        )}
      </div>
    </section>
  );
}
