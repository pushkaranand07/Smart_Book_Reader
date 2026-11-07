import { useEffect, useRef, type KeyboardEvent } from "react";
import type { Page } from "../api/types";
import { padPage, pluralize } from "../lib/text";

interface PageRailProps {
  pages: Page[];
  selectedPage: number | null;
  onSelect: (pageNumber: number) => void;
}

const PAGE_JUMP = 10;

/**
 * Page list as an ARIA listbox with a roving tabindex: one Tab stop, Arrow keys / Home / End / PageUp /
 * PageDown move the selection (selection follows focus).
 */
export function PageRail({ pages, selectedPage, onSelect }: PageRailProps) {
  const optionRefs = useRef(new Map<number, HTMLDivElement>());
  const keyboardMove = useRef(false);

  useEffect(() => {
    if (selectedPage === null) return;
    const option = optionRefs.current.get(selectedPage);
    if (!option) return;
    option.scrollIntoView({ block: "nearest" });
    if (keyboardMove.current) {
      keyboardMove.current = false;
      option.focus();
    }
  }, [selectedPage]);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (pages.length === 0) return;
    const current = Math.max(
      0,
      pages.findIndex((page) => page.page_number === selectedPage),
    );
    const last = pages.length - 1;
    const moves: Record<string, number> = {
      ArrowDown: current + 1,
      ArrowRight: current + 1,
      ArrowUp: current - 1,
      ArrowLeft: current - 1,
      Home: 0,
      End: last,
      PageDown: current + PAGE_JUMP,
      PageUp: current - PAGE_JUMP,
    };
    const target = moves[event.key];
    if (target === undefined) return;
    event.preventDefault();
    const next = pages[Math.min(last, Math.max(0, target))];
    if (!next) return;
    keyboardMove.current = true;
    if (next.page_number === selectedPage) {
      optionRefs.current.get(next.page_number)?.focus();
      keyboardMove.current = false;
    } else {
      onSelect(next.page_number);
    }
  };

  const selectedExists = pages.some((page) => page.page_number === selectedPage);

  return (
    <nav className="page-rail" aria-labelledby="page-rail-heading">
      <div className="section-label">
        <h2 id="page-rail-heading" className="section-label__title">
          Pages
        </h2>
        <span>{pages.length}</span>
      </div>
      {pages.length === 0 ? (
        <p className="muted-note">This book has no pages.</p>
      ) : (
        <div
          className="page-list"
          role="listbox"
          aria-labelledby="page-rail-heading"
          aria-orientation="vertical"
          // Focus lives on the options (roving tabindex); -1 keeps the container out of the Tab order.
          tabIndex={-1}
          onKeyDown={onKeyDown}
        >
          {pages.map((page, index) => {
            const selected = page.page_number === selectedPage;
            const tabbable = selected || (!selectedExists && index === 0);
            const figures = page.figures.length;
            return (
              <div
                key={page.page_number}
                ref={(node) => {
                  if (node) optionRefs.current.set(page.page_number, node);
                  else optionRefs.current.delete(page.page_number);
                }}
                id={`page-option-${page.page_number}`}
                role="option"
                aria-selected={selected}
                tabIndex={tabbable ? 0 : -1}
                className={`page-item ${selected ? "active" : ""}`}
                onClick={() => {
                  onSelect(page.page_number);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelect(page.page_number);
                  }
                }}
              >
                <span className="page-item__number" aria-hidden="true">
                  {padPage(page.page_number)}
                </span>
                <div>
                  <strong>
                    <span className="visually-hidden">Page {page.page_number}, </span>
                    {page.page_type}
                  </strong>
                  <small>
                    {pluralize(page.word_count, "word")}
                    {figures > 0 && ` · ${pluralize(figures, "visual")}`}
                    {page.status !== "ok" && ` · ${page.status}`}
                  </small>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </nav>
  );
}
