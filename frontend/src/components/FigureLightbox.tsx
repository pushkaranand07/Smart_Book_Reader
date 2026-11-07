import { useRef } from "react";
import { apiUrl } from "../api/config";
import type { Figure } from "../api/types";
import { useFocusTrap } from "../hooks/useFocusTrap";
import { figureAlt } from "../lib/text";
import "../styles/lightbox.css";

interface FigureLightboxProps {
  figure: Figure;
  onClose: () => void;
}

/** Modal figure viewer (lazy-loaded). Esc closes, focus is trapped and returned to the opener. */
export default function FigureLightbox({ figure, onClose }: FigureLightboxProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  useFocusTrap(dialogRef, onClose);
  const label = figure.label.trim() || "Figure";

  return (
    <div
      className="image-lightbox"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      role="presentation"
    >
      <div
        ref={dialogRef}
        className="image-lightbox__dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="lightbox-title"
        tabIndex={-1}
      >
        <button className="lightbox-close" type="button" onClick={onClose} aria-label="Close enlarged image">
          ×
        </button>
        <img src={apiUrl(figure.image_url)} alt={figureAlt(figure)} />
        <div className="image-lightbox__caption">
          <h2 id="lightbox-title">
            {label} <span className="image-lightbox__page">· page {figure.page_number}</span>
          </h2>
          <p>{figure.caption ?? "No caption"}</p>
        </div>
      </div>
    </div>
  );
}
