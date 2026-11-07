import type { BookDetail, Figure } from "../api/types";

const HEADING_PATTERNS = [/^(chapter|section)\b/i, /^\d+(?:\.\d+)*\s+[A-Z]/, /^[A-Z][A-Z\s&-]{5,}$/];

/** Rejoin OCR/PDF line fragments into paragraphs, keeping short heading-like lines on their own. */
export function cleanExtractedText(text: string | null | undefined): string {
  const lines = (text ?? "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && line.toLowerCase() !== "n");
  const result: string[] = [];
  let paragraph: string[] = [];

  const flush = () => {
    if (paragraph.length) {
      result.push(paragraph.join(" "));
      paragraph = [];
    }
  };

  for (const line of lines) {
    if (result.at(-1) === line || paragraph.at(-1) === line) continue;
    const isHeading = line.length < 70 && HEADING_PATTERNS.some((pattern) => pattern.test(line));
    if (isHeading) {
      flush();
      result.push(line);
    } else {
      paragraph.push(line);
    }
  }
  flush();
  return result.join("\n\n");
}

export function buildFullText(book: BookDetail): string {
  return book.pages
    .map((page) => `--- PAGE ${page.page_number} (${page.page_type}) ---\n\n${cleanExtractedText(page.text)}`)
    .join("\n\n");
}

export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

export function padPage(pageNumber: number): string {
  return String(pageNumber).padStart(2, "0");
}

export function formatDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function figureAlt(figure: Figure): string {
  const label = figure.label.trim() || `Figure on page ${figure.page_number}`;
  return figure.caption ? `${label}: ${figure.caption}` : label;
}

export function downloadName(filename: string): string {
  return `${filename.replace(/\.pdf$/i, "") || "book"}_full.txt`;
}
