import { describeError } from "../api/errors";

interface ErrorBannerProps {
  error: unknown;
  title?: string;
  onRetry?: () => void;
  onDismiss?: () => void;
}

/** Renders a typed API error `{code, message, request_id}` (or any thrown value) as an alert. */
export function ErrorBanner({ error, title, onRetry, onDismiss }: ErrorBannerProps) {
  const { code, message, requestId } = describeError(error);
  return (
    <div className="error-banner" role="alert">
      <div className="error-banner__body">
        {title && <strong className="error-banner__title">{title}</strong>}
        <p className="error-banner__message">{message}</p>
        <p className="error-banner__meta">
          <span>Code: {code}</span>
          {requestId && <span>Request ID: {requestId}</span>}
        </p>
      </div>
      {(onRetry ?? onDismiss) && (
        <div className="error-banner__actions">
          {onRetry && (
            <button type="button" className="text-button" onClick={onRetry}>
              Try again
            </button>
          )}
          {onDismiss && (
            <button type="button" className="text-button" onClick={onDismiss}>
              Dismiss
            </button>
          )}
        </div>
      )}
    </div>
  );
}
