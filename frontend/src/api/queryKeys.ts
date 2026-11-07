export const queryKeys = {
  health: ["health"] as const,
  books: ["books"] as const,
  book: (bookId: string) => ["books", bookId] as const,
  job: (jobId: string) => ["jobs", jobId] as const,
};
