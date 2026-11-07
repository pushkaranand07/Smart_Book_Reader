import { useState } from "react";
import { apiUrl } from "../api/config";
import type { Figure } from "../api/types";
import { figureAlt } from "../lib/text";

interface FigureCardProps {
  figure: Figure;
  highlighted?: boolean;
  variant?: "strip" | "answer";
  onOpen: (figure: Figure) => void;
}

/** Thumbnail + label + caption (or a neutral "No caption") for one detected figure. */
export function FigureCard({ figure, highlighted = false, variant = "strip", onOpen }: FigureCardProps) {
  const [failed, setFailed] = useState(false);
  const label = figure.label.trim() || "Figure";

  return (
    <figure
      id={`figure-${figure.figure_id}`}
      className={`figure-card figure-card--${variant} ${highlighted ? "figure-card--highlighted" : ""}`}
      data-figure-id={figure.figure_id}
    >
      <button
        type="button"
        className="figure-button"
        onClick={() => {
          onOpen(figure);
        }}
        aria-label={`Enlarge ${label}`}
      >
        {failed ? (
          <span className="figure-button__fallback">Image unavailable</span>
        ) : (
          <img
            src={apiUrl(figure.image_url)}
            alt={figureAlt(figure)}
            loading="lazy"
            decoding="async"
            onError={() => {
              setFailed(true);
            }}
          />
        )}
      </button>
      <figcaption>
        <strong>{label}</strong>
        {figure.caption ? (
          <span className="figure-card__caption">{figure.caption}</span>
        ) : (
          <span className="figure-card__caption figure-card__caption--none">No caption</span>
        )}
      </figcaption>
    </figure>
  );
}
