import type { Citation } from "../api/types";

interface CitationsProps {
  citations: Citation[];
  figureLabel: (figureId: string) => string;
  onCitationSelect: (citation: Citation) => void;
  onFigureSelect: (figureId: string, fallbackPage: number) => void;
}

const SNIPPET_LIMIT = 180;

function shorten(text: string): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > SNIPPET_LIMIT ? `${clean.slice(0, SNIPPET_LIMIT - 1)}…` : clean;
}

export function Citations({ citations, figureLabel, onCitationSelect, onFigureSelect }: CitationsProps) {
  if (citations.length === 0) {
    return <p className="muted-note citations__empty">No citations were returned for this answer.</p>;
  }
  return (
    <div className="citations">
      <h3 className="copy-label">Sources</h3>
      <ol className="citations__list">
        {citations.map((citation, index) => (
          <li key={`${citation.chunk_id}-${index}`} className="citation">
            <button
              type="button"
              className="citation__button"
              data-page-number={citation.page_number}
              onClick={() => {
                onCitationSelect(citation);
              }}
            >
              <span className="citation__page">
                <span aria-hidden="true">[{index + 1}] </span>
                Page {citation.page_number}
              </span>
              {citation.section && <span className="citation__section">{citation.section}</span>}
              <span className="citation__snippet">{shorten(citation.snippet)}</span>
            </button>
            {citation.figure_ids.length > 0 && (
              <ul className="citation__figures" aria-label={`Figures cited on page ${citation.page_number}`}>
                {citation.figure_ids.map((figureId) => (
                  <li key={figureId}>
                    <button
                      type="button"
                      className="chip-button"
                      onClick={() => {
                        onFigureSelect(figureId, citation.page_number);
                      }}
                    >
                      Show {figureLabel(figureId)}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
