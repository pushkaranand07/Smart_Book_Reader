import { describe, expect, it } from "vitest";
import { bookDetail, figure, uncaptionedFigure } from "../test/fixtures";
import { buildFullText, cleanExtractedText, downloadName, figureAlt, pluralize } from "./text";

describe("text helpers", () => {
  it("joins wrapped lines into paragraphs and keeps headings separate", () => {
    const raw = "CHAPTER ONE\nPlants make\nfood.\nn\nfood.\n1.2 Leaves\nLeaves are green.";
    expect(cleanExtractedText(raw)).toBe("CHAPTER ONE\n\nPlants make food.\n\n1.2 Leaves\n\nLeaves are green.");
    expect(cleanExtractedText(null)).toBe("");
  });

  it("builds the full text with page markers", () => {
    const text = buildFullText(bookDetail);
    expect(text).toContain("--- PAGE 1 (Digital) ---");
    expect(text).toContain("--- PAGE 3 (Scanned) ---");
  });

  it("formats labels", () => {
    expect(pluralize(1, "page")).toBe("1 page");
    expect(pluralize(3, "visual")).toBe("3 visuals");
    expect(figureAlt(figure)).toBe("Figure 2.1: Structure of a plant cell");
    expect(figureAlt(uncaptionedFigure)).toBe("Figure 3.1");
    expect(figureAlt({ ...uncaptionedFigure, label: " " })).toBe("Figure on page 3");
    expect(downloadName("Biology.PDF")).toBe("Biology_full.txt");
  });
});
