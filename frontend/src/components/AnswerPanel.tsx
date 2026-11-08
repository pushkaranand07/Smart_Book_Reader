import { lazy, Suspense } from "react";
import type { Answer, Citation, Figure } from "../api/types";
import { Citations } from "./Citations";
import { FigureCard } from "./FigureCard";

const AnswerMarkdown = lazy(() => import("./AnswerMarkdown"));

interface AnswerPanelProps {
  answer: Answer;
  figureLabel: (figureId: string) => string;
  onCitationSelect: (citation: Citation) => void;
  onFigureSelect: (figureId: string, fallbackPage: number) => void;
  onOpenFigure: (figure: Figure) => void;
}

const MODE_LABEL: Record<Answer["mode"], string> = {
  offline: "Offline",
  gemini: "Gemini",
  azure_openai: "Azure OpenAI",
};
const MAX_ANSWER_FIGURES = 3;

export function AnswerPanel({ answer, figureLabel, onCitationSelect, onFigureSelect, onOpenFigure }: AnswerPanelProps) {
  const confidence = Math.round(Math.min(1, Math.max(0, answer.confidence)) * 100);
  const figures = answer.figures.slice(0, MAX_ANSWER_FIGURES);

  return (
    <div className="answer-layout" data-testid="answer">
      <article className="answer-main" aria-labelledby="answer-heading">
        <div className="answer-meta">
          <h3 id="answer-heading" className="copy-label">
            Answer
          </h3>
          <span className={`mode-badge mode-badge--${answer.mode}`} title="Answer mode">
            Mode: {MODE_LABEL[answer.mode]}
          </span>
          <span className="answer-confidence">Confidence {confidence}%</span>
        </div>

        {!answer.is_sufficient && (
          <div className="not-covered" role="note" data-testid="not-covered">
            <strong>Not covered in this book</strong>
            <p>
              The book does not appear to answer this question. Try rephrasing it or asking about a topic from its
              pages.
            </p>
          </div>
        )}

        {answer.answer.trim() && (
          <Suspense fallback={<div className="answer-text answer-text--plain">{answer.answer}</div>}>
            <AnswerMarkdown markdown={answer.answer} />
          </Suspense>
        )}

        {answer.extractive_answer && (
          <div className="extractive">
            <h4 className="copy-label">Exact passage from the book</h4>
            <blockquote className="extractive__quote">{answer.extractive_answer}</blockquote>
          </div>
        )}

        <Citations
          citations={answer.citations}
          figureLabel={figureLabel}
          onCitationSelect={onCitationSelect}
          onFigureSelect={onFigureSelect}
        />
      </article>

      {figures.length > 0 && (
        <aside className="answer-figures" aria-labelledby="answer-figures-heading">
          <h3 id="answer-figures-heading" className="copy-label">
            Associated visuals
          </h3>
          {figures.map((figure) => (
            <FigureCard
              key={`${figure.figure_key}:${figure.image_url}`}
              figure={figure}
              variant="answer"
              onOpen={onOpenFigure}
            />
          ))}
        </aside>
      )}
    </div>
  );
}
