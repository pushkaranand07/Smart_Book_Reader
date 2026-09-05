import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import Markdown from "react-markdown";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");

function assetUrl(path) {
  if (!path || path.startsWith("http")) return path;
  return path.startsWith("/") ? `${API}${path}` : path;
}

function cleanExtractedText(text) {
  const lines = String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && line.toLowerCase() !== "n");
  const result = [];
  let paragraph = [];

  const flush = () => {
    if (paragraph.length) {
      result.push(paragraph.join(" "));
      paragraph = [];
    }
  };

  for (const line of lines) {
    if (result[result.length - 1] === line || paragraph[paragraph.length - 1] === line) continue;
    const heading = line.length < 70 && (
      /^(chapter|section)\b/i.test(line) ||
      /^\d+(?:\.\d+)*\s+[A-Z]/.test(line) ||
      /^[A-Z][A-Z\s&-]{5,}$/.test(line)
    );
    if (heading) {
      flush();
      result.push(line);
    } else {
      paragraph.push(line);
    }
  }
  flush();
  return result.join("\n\n");
}

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || "The server could not complete that request.");
  return body;
}

function Metric({ label, value, tone = "ink" }) {
  return <div className={`metric metric-${tone}`}><span>{label}</span><strong>{value}</strong></div>;
}

function App() {
  const [book, setBook] = useState(null);
  const [question, setQuestion] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [answer, setAnswer] = useState(null);
  const [selectedPage, setSelectedPage] = useState(0);
  const [busy, setBusy] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");

  const pages = book?.normalized_document?.pages || book?.pages || [];
  const page = pages[selectedPage];
  const figureCount = pages.reduce((total, item) => total + (item.figures?.length || 0), 0);
  const fullText = pages.map((item) => `--- PAGE ${item.page_number} (${item.page_type}) ---\n\n${cleanExtractedText(item.text)}`).join("\n\n");

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true); setError(""); setAnswer(null);
    const form = new FormData(); form.append("file", file);
    try { setBook(await request("/api/books/upload", { method: "POST", body: form })); setSelectedPage(0); }
    catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  async function ask(event) {
    event.preventDefault();
    if (!book || !question.trim()) return;
    setAsking(true); setError("");
    try { setAnswer(await request(`/api/books/${book.book_id}/questions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, api_key: apiKey || null }) })); }
    catch (err) { setError(err.message); }
    finally { setAsking(false); }
  }

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark">SR</div><div><strong>Smart Reader</strong><span>Visual document intelligence</span></div></div>
      <div className="topbar-status"><span className="status-dot" /> Florence-2 connected <span className="divider" /> Local-first</div>
    </header>

    <main className="workspace">
      <section className="intro-row">
        <div><p className="eyebrow">Document workspace</p><h1>Read the page,<br /><em>see the meaning.</em></h1><p className="lede">Upload a textbook or paper to search its words, figures, and visual structure in one calm workspace.</p></div>
        <div className="upload-panel"><div className="upload-icon">+</div><div><strong>{busy ? "Analyzing your document" : "Open a PDF"}</strong><span>{busy ? "OCR, figures, and page structure are being prepared" : "Digital and scanned pages supported"}</span></div><label className="upload-button">{busy ? "Working..." : "Choose file"}<input type="file" accept="application/pdf" onChange={upload} disabled={busy} /></label></div>
      </section>

      {error && <div className="error-banner">{error}</div>}

      {!book && !busy && <section className="empty-state"><div className="empty-number">01</div><div><h2>Your document starts here</h2><p>Choose a PDF to activate page analysis, grounded search, and figure discovery.</p></div><div className="empty-line" /></section>}
      {busy && <section className="loading-state"><div className="loader" /><div><h2>Preparing your reading room</h2><p>The first run may take a moment while the vision and OCR engines inspect each page.</p></div></section>}

      {book && <>
        <section className="metrics-grid"><Metric label="Total pages" value={book.total_pages} /><Metric label="Digital pages" value={book.digital_pages} tone="green" /><Metric label="Scanned pages" value={book.scanned_pages} tone="amber" /><Metric label="Figures found" value={figureCount} tone="blue" /></section>

        <section className="reader-grid">
          <aside className="page-rail"><div className="section-label">Pages <span>{pages.length}</span></div><div className="page-list">{pages.map((item, index) => <button key={item.page_number} className={index === selectedPage ? "page-item active" : "page-item"} onClick={() => setSelectedPage(index)}><span>{String(item.page_number).padStart(2, "0")}</span><div><strong>{item.page_type}</strong><small>{item.word_count || 0} words {item.figures?.length ? `· ${item.figures.length} visual${item.figures.length > 1 ? "s" : ""}` : ""}</small></div></button>)}</div></aside>
          <article className="page-view"><div className="page-view-head"><div><p className="eyebrow">Page {page?.page_number}</p><h2>{page?.page_type === "Scanned" ? "Scanned page" : "Document page"}</h2></div><span className={page?.page_type === "Scanned" ? "type-pill amber" : "type-pill green"}>{page?.page_type}</span></div><div className="page-content"><div className="page-copy"><p className="copy-label">Extracted text</p><p>{page?.text || "No text was detected on this page."}</p></div>{page?.figures?.length > 0 && <div className="page-figures"><p className="copy-label">Visual regions</p><div className="figure-strip">{page.figures.map((figure, index) => <img key={figure.figure_id || index} src={assetUrl(figure.image_path)} alt={figure.figure_label || "Detected visual"} />)}</div></div>}</div></article>
        </section>

        <section className="qa-section"><div className="qa-heading"><div><p className="eyebrow">Grounded inquiry</p><h2>Ask this book</h2></div><span>Clear explanations from your document</span></div><form className="question-form" onSubmit={ask}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What would you like to understand?" /><input className="api-key-input" type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="Optional Gemini API key" autoComplete="off" /><button type="submit" disabled={asking || !question.trim()}>{asking ? "Searching..." : "Ask"}<span>↗</span></button></form>{answer && <div className="answer-layout"><div className="answer-main"><p className="copy-label">Answer</p><div className="answer-text"><Markdown>{answer.answer}</Markdown></div></div>{answer.figures?.length > 0 && <div className="answer-figures"><p className="copy-label">Associated visuals</p>{answer.figures.slice(0, 3).map((figure, index) => <figure key={figure.figure_id || index}><img src={assetUrl(figure.image_path)} alt={figure.figure_label || "Associated visual"} /><figcaption>{figure.figure_label || "Figure"}{figure.caption ? ` · ${figure.caption}` : ""}</figcaption></figure>)}</div>}</div>}</section>

        <section className="text-section"><div className="qa-heading"><div><p className="eyebrow">Reference</p><h2>Full text</h2></div><a className="text-link" href={`data:text/plain;charset=utf-8,${encodeURIComponent(fullText)}`} download={`${book.filename.replace(/\.pdf$/i, "")}_full.txt`}>Download .txt ↗</a></div><pre>{fullText}</pre></section>
      </>}
    </main>
    <footer><span>SMART READER / {book?.filename || "NO DOCUMENT LOADED"}</span><span>Pipeline ready for exploration</span></footer>
  </div>;
}

createRoot(document.getElementById("root")).render(<StrictMode><App /></StrictMode>);
