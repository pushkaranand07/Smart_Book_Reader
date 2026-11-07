import { useQuery } from "@tanstack/react-query";
import { getBook, listBooks } from "../api/client";
import { queryKeys } from "../api/queryKeys";

export function useBooks() {
  return useQuery({
    queryKey: queryKeys.books,
    queryFn: ({ signal }) => listBooks(signal),
    select: (data) => [...data.books].sort((a, b) => b.created_at.localeCompare(a.created_at)),
  });
}

/** Book detail. Figure image URLs are signed and short-lived, so the detail is refreshed every few minutes. */
export function useBook(bookId: string | null, refreshMs = 4 * 60_000) {
  return useQuery({
    queryKey: queryKeys.book(bookId ?? ""),
    queryFn: ({ signal }) => getBook(bookId ?? "", signal),
    enabled: bookId !== null,
    staleTime: refreshMs,
    refetchInterval: refreshMs,
  });
}
