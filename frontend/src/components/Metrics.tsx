import type { BookDetail } from "../api/types";

type Tone = "ink" | "green" | "amber" | "red";

function Metric({ label, value, tone = "ink" }: { label: string; value: number; tone?: Tone }) {
  return (
    <div className={`metric metric-${tone}`}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

export function Metrics({ book }: { book: BookDetail }) {
  const needsReview = book.pages.filter((page) => page.status !== "ok").length;
  return (
    <section aria-label="Document summary">
      <dl className="metrics-grid">
        <Metric label="Total pages" value={book.total_pages} />
        <Metric label="Digital pages" value={book.digital_pages} tone="green" />
        <Metric label="Scanned pages" value={book.scanned_pages} tone="amber" />
        <Metric label="Needs review" value={needsReview} tone="red" />
      </dl>
    </section>
  );
}
