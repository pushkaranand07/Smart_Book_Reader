import { useMemo } from "react";
import type { BookDetail } from "../api/types";
import { buildFullText, downloadName } from "../lib/text";
import "../styles/fulltext.css";

/** Full extracted text with a .txt download (lazy-loaded; can be large for long books). */
export default function FullTextPanel({ book }: { book: BookDetail }) {
  const fullText = useMemo(() => buildFullText(book), [book]);

  const download = () => {
    const url = URL.createObjectURL(new Blob([fullText], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = downloadName(book.filename);
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div className="qa-heading">
        <div>
          <p className="eyebrow">Reference</p>
          <h2 id="full-text-heading">Full text</h2>
        </div>
        <button type="button" className="text-link" onClick={download}>
          Download .txt <span aria-hidden="true">↗</span>
        </button>
      </div>
      {/* Scrollable region must be focusable so keyboard users can scroll it. */}
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex */}
      <pre className="full-text" tabIndex={0} aria-labelledby="full-text-heading">
        {fullText || "No text was extracted from this book."}
      </pre>
    </>
  );
}
