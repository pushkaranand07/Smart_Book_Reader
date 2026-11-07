import type { BookSummary } from "../api/types";
import { useBooks } from "../hooks/useBooks";
import { formatDate, pluralize } from "../lib/text";
import { ErrorBanner } from "./ErrorBanner";

interface BookListProps {
  activeBookId: string | null;
  onOpen: (book: BookSummary) => void;
}

/** The caller's earlier uploads (`GET /api/books`), so a book can be reopened without re-uploading. */
export function BookList({ activeBookId, onOpen }: BookListProps) {
  const books = useBooks();

  return (
    <section className="book-list" aria-labelledby="book-list-heading">
      <div className="section-label">
        <h2 id="book-list-heading" className="section-label__title">
          Your books
        </h2>
        {books.data && <span>{books.data.length}</span>}
      </div>
      {books.isPending && <p className="muted-note">Loading your books...</p>}
      {books.isError && (
        <ErrorBanner
          error={books.error}
          title="Could not load your books"
          onRetry={() => {
            void books.refetch();
          }}
        />
      )}
      {books.data?.length === 0 && <p className="muted-note">No books yet. Upload a PDF to get started.</p>}
      {books.data && books.data.length > 0 && (
        <ul className="book-list__items">
          {books.data.map((book) => {
            const active = book.book_id === activeBookId;
            return (
              <li key={book.book_id}>
                <button
                  type="button"
                  className={`book-item ${active ? "book-item--active" : ""}`}
                  aria-current={active ? "true" : undefined}
                  onClick={() => {
                    onOpen(book);
                  }}
                >
                  <strong>{book.filename}</strong>
                  <small>
                    {pluralize(book.total_pages, "page")} · {formatDate(book.created_at)} · {book.status}
                  </small>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
