import { useMutation } from "@tanstack/react-query";
import { askQuestion } from "../api/client";

export function useAskQuestion(bookId: string | null) {
  return useMutation({
    mutationKey: ["ask", bookId],
    mutationFn: (question: string) => {
      if (!bookId) return Promise.reject(new Error("Open a book before asking a question."));
      return askQuestion(bookId, question);
    },
  });
}
