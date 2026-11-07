import { useId, useState, type SubmitEvent } from "react";

interface QuestionFormProps {
  pending: boolean;
  onAsk: (question: string) => void;
}

export function QuestionForm({ pending, onAsk }: QuestionFormProps) {
  const [question, setQuestion] = useState("");
  const inputId = useId();
  const trimmed = question.trim();

  const submit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (trimmed && !pending) onAsk(trimmed);
  };

  return (
    <form className="question-form" onSubmit={submit} role="search" aria-label="Ask this book">
      <label htmlFor={inputId} className="visually-hidden">
        Your question about this book
      </label>
      <input
        id={inputId}
        value={question}
        onChange={(event) => {
          setQuestion(event.target.value);
        }}
        placeholder="What would you like to understand?"
        autoComplete="off"
        maxLength={2000}
      />
      <button type="submit" disabled={pending || !trimmed}>
        {pending ? "Searching..." : "Ask"}
        <span aria-hidden="true">↗</span>
      </button>
    </form>
  );
}
