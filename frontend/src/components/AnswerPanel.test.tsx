import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { answer, insufficientAnswer } from "../test/fixtures";
import { AnswerPanel } from "./AnswerPanel";

// The markdown renderer is a lazy chunk; its first import can be slow under parallel test load.
const LAZY = { timeout: 5000 };

function renderPanel(overrides: Partial<Parameters<typeof AnswerPanel>[0]> = {}) {
  const props = {
    answer,
    figureLabel: (id: string) => (id === "fig-2-1" ? "Figure 2.1" : id),
    onCitationSelect: vi.fn(),
    onFigureSelect: vi.fn(),
    onOpenFigure: vi.fn(),
    ...overrides,
  };
  render(<AnswerPanel {...props} />);
  return props;
}

describe("AnswerPanel", () => {
  it("renders markdown answer, extractive answer, mode and confidence", async () => {
    renderPanel();
    // Lazy markdown renderer turns **cell wall** into <strong>.
    expect(await screen.findByText("cell wall", { selector: "strong" }, LAZY)).toBeInTheDocument();
    expect(screen.getByText("Exact passage from the book")).toBeInTheDocument();
    expect(screen.getByText("a cell wall and chloroplasts")).toBeInTheDocument();
    expect(screen.getByText("Mode: Offline")).toBeInTheDocument();
    expect(screen.getByText("Confidence 82%")).toBeInTheDocument();
    expect(screen.queryByTestId("not-covered")).not.toBeInTheDocument();
  });

  it("calls onCitationSelect with the clicked citation", async () => {
    const user = userEvent.setup();
    const props = renderPanel();
    await user.click(screen.getByRole("button", { name: /Page 3/ }));
    expect(props.onCitationSelect).toHaveBeenCalledWith(answer.citations[1]);
  });

  it("jumps to a cited figure via its figure id", async () => {
    const user = userEvent.setup();
    const props = renderPanel();
    await user.click(screen.getByRole("button", { name: "Show Figure 2.1" }));
    expect(props.onFigureSelect).toHaveBeenCalledWith("fig-2-1", 2);
  });

  it("shows a clear not-covered state when the answer is insufficient", async () => {
    renderPanel({ answer: insufficientAnswer });
    expect(screen.getByTestId("not-covered")).toHaveTextContent("Not covered in this book");
    expect(screen.getByText("Mode: Gemini")).toBeInTheDocument();
    expect(screen.getByText("No citations were returned for this answer.")).toBeInTheDocument();
    expect(screen.queryByText("Exact passage from the book")).not.toBeInTheDocument();
    expect(await screen.findByText(/does not discuss quantum chromodynamics/, {}, LAZY)).toBeInTheDocument();
  });

  it("opens associated figures in the viewer", async () => {
    const user = userEvent.setup();
    const props = renderPanel();
    await user.click(screen.getByRole("button", { name: "Enlarge Figure 2.1" }));
    expect(props.onOpenFigure).toHaveBeenCalledWith(answer.figures[0]);
  });
});
