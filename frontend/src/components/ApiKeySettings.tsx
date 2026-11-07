import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState, type SubmitEvent, type KeyboardEvent } from "react";
import { useApiKey } from "../hooks/useApiKey";

/** Settings popover for the API key. Opens automatically when the server answers 401. */
export function ApiKeySettings() {
  const { apiKey, keyRequired, setApiKey, dismissPrompt } = useApiKey();
  const queryClient = useQueryClient();
  const [manualOpen, setManualOpen] = useState(false);
  const [draft, setDraft] = useState(apiKey);
  const open = manualOpen || keyRequired;
  const popoverId = useId();
  const inputId = useId();
  const titleId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  const close = (returnFocus = true) => {
    setManualOpen(false);
    dismissPrompt();
    if (returnFocus) toggleRef.current?.focus();
  };

  const toggle = () => {
    if (open) {
      close(false);
    } else {
      setDraft(apiKey);
      setManualOpen(true);
    }
  };

  const save = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    setApiKey(draft);
    setManualOpen(false);
    toggleRef.current?.focus();
    void queryClient.invalidateQueries();
  };

  const clear = () => {
    setDraft("");
    setApiKey("");
    void queryClient.invalidateQueries();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      close();
    }
  };

  return (
    <div className="settings">
      <button
        ref={toggleRef}
        type="button"
        className="settings__toggle"
        aria-expanded={open}
        aria-controls={popoverId}
        onClick={toggle}
      >
        <span className={`key-dot ${apiKey ? "key-dot--set" : ""}`} aria-hidden="true" />
        API key{apiKey ? " set" : ""}
      </button>
      {open && (
        // The popover is a non-modal dialog; Escape closes it and returns focus to the toggle.
        // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions
        <div id={popoverId} className="settings__popover" role="dialog" aria-labelledby={titleId} onKeyDown={onKeyDown}>
          <h2 id={titleId} className="settings__title">
            API access
          </h2>
          {keyRequired && (
            <p className="settings__warning" role="alert">
              The server rejected the request (401). Enter a valid API key to continue.
            </p>
          )}
          <form onSubmit={save}>
            <label htmlFor={inputId} className="field-label">
              API key
            </label>
            <input
              ref={inputRef}
              id={inputId}
              className="text-input"
              type="password"
              autoComplete="off"
              spellCheck={false}
              value={draft}
              onChange={(event) => {
                setDraft(event.target.value);
              }}
            />
            <p className="settings__hint">Stored in this tab only (sessionStorage) and sent as X-API-Key.</p>
            <div className="settings__actions">
              <button type="submit" className="primary-button">
                Save key
              </button>
              <button type="button" className="text-button" onClick={clear} disabled={!apiKey && !draft}>
                Clear
              </button>
              <button
                type="button"
                className="text-button"
                onClick={() => {
                  close();
                }}
              >
                Close
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
