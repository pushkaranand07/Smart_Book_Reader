import { useId, type ChangeEvent } from "react";

interface UploadPanelProps {
  busy: boolean;
  onFile: (file: File) => void;
}

export function UploadPanel({ busy, onFile }: UploadPanelProps) {
  const inputId = useId();
  const hintId = useId();

  const onChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    // Reset so choosing the same file again still fires a change event.
    event.target.value = "";
    if (file) onFile(file);
  };

  return (
    <div className="upload-panel">
      <div className="upload-icon" aria-hidden="true">
        +
      </div>
      <div>
        <strong>{busy ? "Analyzing your document" : "Open a PDF"}</strong>
        <span id={hintId}>
          {busy ? "OCR, figures, and page structure are being prepared" : "Digital and scanned pages supported"}
        </span>
      </div>
      {/* The input stays in the tab order (visually hidden); its focus ring is drawn on the label after it. */}
      <input
        id={inputId}
        className="visually-hidden upload-input"
        type="file"
        accept="application/pdf,.pdf"
        aria-describedby={hintId}
        disabled={busy}
        onChange={onChange}
      />
      <label className={`upload-button ${busy ? "upload-button--disabled" : ""}`} htmlFor={inputId}>
        {busy ? "Working..." : "Choose file"}
        <span className="visually-hidden"> (PDF document to upload)</span>
      </label>
    </div>
  );
}
