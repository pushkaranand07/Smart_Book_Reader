import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { figure, uncaptionedFigure } from "../test/fixtures";
import { FigureCard } from "./FigureCard";

describe("FigureCard", () => {
  it("shows the caption and uses the signed image URL", () => {
    render(<FigureCard figure={figure} onOpen={vi.fn()} />);
    expect(screen.getByText("Structure of a plant cell")).toBeInTheDocument();
    const image = screen.getByRole("img");
    expect(image).toHaveAttribute("src", figure.image_url);
    expect(image).toHaveAttribute("alt", "Figure 2.1: Structure of a plant cell");
  });

  it('shows a neutral "No caption" when caption is null', () => {
    render(<FigureCard figure={uncaptionedFigure} onOpen={vi.fn()} />);
    expect(screen.getByText("No caption")).toBeInTheDocument();
    expect(screen.getByRole("img")).toHaveAttribute("alt", "Figure 3.1");
  });

  it("falls back to a placeholder when the image fails to load", () => {
    render(<FigureCard figure={figure} onOpen={vi.fn()} />);
    fireEvent.error(screen.getByRole("img"));
    expect(screen.getByText("Image unavailable")).toBeInTheDocument();
  });
});
