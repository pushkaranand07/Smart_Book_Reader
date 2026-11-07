import { useEffect, useRef } from "react";
import type { Figure, Page } from "../api/types";
import { FigureCard } from "./FigureCard";

interface PageViewProps {
  page: Page | undefined;
  highlightedFigureId: string | null;
  /** Incremented when a citation jumps here: scroll into view, move focus and flash the highlight. */
  focusRequest: number;
  onOpenFigure: (figure: Figure) => void;
}

function pillTone(page: Page): string {
  if (page.status === "failed") return "red";
  return page.page_type === "Scanned" ? "amber" : "green";
}

export function PageView({ page, highlightedFigureId, focusRequest, onOpenFigure }: PageViewProps) {
  const articleRef = useRef<HTMLElement>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    if (focusRequest === 0) return;
    const article = articleRef.current;
    if (!article) return;
    const figure = highlightedFigureId
      ? (Array.from(article.querySelectorAll<HTMLElement>("[data-figure-id]")).find(
          (node) => node.dataset.figureId === highlightedFigureId,
        ) ?? null)
      : null;
    (figure ?? article).scrollIntoView({ behavior: "smooth", block: figure ? "center" : "start" });
    headingRef.current?.focus({ preventScroll: true });
    article.classList.remove("page-view--flash");
    // Force a reflow so the flash animation restarts on repeated jumps.
    article.getBoundingClientRect();
    article.classList.add("page-view--flash");
  }, [focusRequest, highlightedFigureId]);

  if (!page) {
    return (
      <article className="page-view" aria-label="Page view">
        <div className="page-content">
          <p className="muted-note">Select a page to read it.</p>
        </div>
      </article>
    );
  }

  return (
    <article
      ref={articleRef}
      className="page-view"
      aria-labelledby="page-view-heading"
      data-testid="page-view"
      data-page-number={page.page_number}
    >
      <div className="page-view-head">
        <div>
          <p className="eyebrow">Page {page.page_number}</p>
          <h2 id="page-view-heading" ref={headingRef} tabIndex={-1}>
            {page.page_type === "Scanned" ? "Scanned page" : "Document page"}
            <span className="visually-hidden"> {page.page_number}</span>
          </h2>
        </div>
        <span className={`type-pill ${pillTone(page)}`}>{page.status === "ok" ? page.page_type : page.status}</span>
      </div>
      <div className="page-content">
        {page.status === "failed" && (
          <div className="error-banner" role="note">
            This page could not be processed. The remaining pages are available.
          </div>
        )}
        {page.errors.length > 0 && (
          <ul className="page-errors" aria-label="Processing notes">
            {page.errors.map((error) => (
              <li key={`${error.code}-${error.message}`}>
                <code>{error.code}</code> {error.message}
              </li>
            ))}
          </ul>
        )}
        <div className="page-copy">
          <h3 className="copy-label">Extracted text</h3>
          <p className="page-copy__text">{page.text || "No text was detected on this page."}</p>
        </div>
        {page.figures.length > 0 && (
          <div className="page-figures">
            <h3 className="copy-label">Visual regions</h3>
            <div className="figure-strip">
              {page.figures.map((figure) => (
                <FigureCard
                  key={`${figure.figure_key}:${figure.image_url}`}
                  figure={figure}
                  highlighted={figure.figure_id === highlightedFigureId}
                  onOpen={onOpenFigure}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </article>
  );
}
