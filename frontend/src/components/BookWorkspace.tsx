import { lazy, Suspense, useCallback, useMemo, useState } from "react";
import type { BookDetail, Citation, Figure } from "../api/types";
import { Metrics } from "./Metrics";
import { PageRail } from "./PageRail";
import { PageView } from "./PageView";
import { QASection } from "./QASection";

const FigureLightbox = lazy(() => import("./FigureLightbox"));
const FullTextPanel = lazy(() => import("./FullTextPanel"));

interface SelectionState {
  page: number | null;
  figureId: string | null;
  focusRequest: number;
}

/** Everything shown for a loaded book. Mount with `key={book_id}` so selection resets per book. */
export function BookWorkspace({ book }: { book: BookDetail }) {
  const [selection, setSelection] = useState<SelectionState>({
    page: book.pages[0]?.page_number ?? null,
    figureId: null,
    focusRequest: 0,
  });
  const [lightboxFigure, setLightboxFigure] = useState<Figure | null>(null);

  const figuresById = useMemo(() => {
    const map = new Map<string, Figure>();
    for (const page of book.pages) for (const figure of page.figures) map.set(figure.figure_id, figure);
    return map;
  }, [book]);

  const page = book.pages.find((item) => item.page_number === selection.page);

  const selectPage = useCallback((pageNumber: number) => {
    setSelection((current) => ({ ...current, page: pageNumber, figureId: null }));
  }, []);

  const jumpTo = useCallback((pageNumber: number, figureId: string | null) => {
    setSelection((current) => ({ page: pageNumber, figureId, focusRequest: current.focusRequest + 1 }));
  }, []);

  const onCitationSelect = useCallback(
    (citation: Citation) => {
      jumpTo(citation.page_number, null);
    },
    [jumpTo],
  );

  const onFigureSelect = useCallback(
    (figureId: string, fallbackPage: number) => {
      jumpTo(figuresById.get(figureId)?.page_number ?? fallbackPage, figureId);
    },
    [figuresById, jumpTo],
  );

  const figureLabel = useCallback(
    (figureId: string) => figuresById.get(figureId)?.label.trim() || `figure ${figureId}`,
    [figuresById],
  );

  return (
    <>
      <Metrics book={book} />

      <section className="reader-grid" aria-label="Pages">
        <PageRail pages={book.pages} selectedPage={selection.page} onSelect={selectPage} />
        <PageView
          page={page}
          highlightedFigureId={selection.figureId}
          focusRequest={selection.focusRequest}
          onOpenFigure={setLightboxFigure}
        />
      </section>

      <QASection
        bookId={book.book_id}
        figureLabel={figureLabel}
        onCitationSelect={onCitationSelect}
        onFigureSelect={onFigureSelect}
        onOpenFigure={setLightboxFigure}
      />

      <section className="text-section" aria-labelledby="full-text-heading">
        <Suspense fallback={<p className="muted-note">Loading full text...</p>}>
          <FullTextPanel book={book} />
        </Suspense>
      </section>

      {lightboxFigure && (
        <Suspense fallback={null}>
          <FigureLightbox
            figure={lightboxFigure}
            onClose={() => {
              setLightboxFigure(null);
            }}
          />
        </Suspense>
      )}
    </>
  );
}
