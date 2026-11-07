import type { Citation, Figure } from "../api/types";
import { useAskQuestion } from "../hooks/useAskQuestion";
import { AnswerPanel } from "./AnswerPanel";
import { ErrorBanner } from "./ErrorBanner";
import { QuestionForm } from "./QuestionForm";

interface QASectionProps {
  bookId: string;
  figureLabel: (figureId: string) => string;
  onCitationSelect: (citation: Citation) => void;
  onFigureSelect: (figureId: string, fallbackPage: number) => void;
  onOpenFigure: (figure: Figure) => void;
}

export function QASection({ bookId, ...handlers }: QASectionProps) {
  const ask = useAskQuestion(bookId);

  return (
    <section className="qa-section" aria-labelledby="qa-heading">
      <div className="qa-heading">
        <div>
          <p className="eyebrow">Grounded inquiry</p>
          <h2 id="qa-heading">Ask this book</h2>
        </div>
        <span>Clear explanations from your document</span>
      </div>
      <QuestionForm
        pending={ask.isPending}
        onAsk={(question) => {
          ask.mutate(question);
        }}
      />
      <div aria-live="polite" aria-busy={ask.isPending} className="qa-results">
        {ask.isPending && <p className="muted-note">Searching the book...</p>}
        {ask.isError && (
          <ErrorBanner
            error={ask.error}
            title="The question could not be answered"
            onDismiss={() => {
              ask.reset();
            }}
          />
        )}
        {ask.data && <AnswerPanel answer={ask.data} {...handlers} />}
      </div>
    </section>
  );
}
