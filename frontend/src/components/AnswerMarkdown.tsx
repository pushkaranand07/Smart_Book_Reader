import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import "../styles/markdown.css";

/** Markdown + LaTeX math renderer. Lazy-loaded: react-markdown and KaTeX are the heaviest dependencies. */
export default function AnswerMarkdown({ markdown }: { markdown: string }) {
  return (
    <div className="answer-text markdown-body">
      <Markdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
        {markdown}
      </Markdown>
    </div>
  );
}
