import type { ReactNode } from "react";

interface EmptyStateProps {
  number?: string;
  title: string;
  children: ReactNode;
}

export function EmptyState({ number = "01", title, children }: EmptyStateProps) {
  return (
    <section className="empty-state" aria-label={title}>
      <div className="empty-number" aria-hidden="true">
        {number}
      </div>
      <div>
        <h2>{title}</h2>
        <p>{children}</p>
      </div>
      <div className="empty-line" aria-hidden="true" />
    </section>
  );
}
