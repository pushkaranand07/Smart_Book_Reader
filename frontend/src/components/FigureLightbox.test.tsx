import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { figure, uncaptionedFigure } from "../test/fixtures";
import FigureLightbox from "./FigureLightbox";

function Harness({ target = figure }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => {
          setOpen(true);
        }}
      >
        Open
      </button>
      {open && (
        <FigureLightbox
          figure={target}
          onClose={() => {
            setOpen(false);
          }}
        />
      )}
    </>
  );
}

describe("FigureLightbox", () => {
  it("moves focus in, traps Tab, closes on Escape and returns focus", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const opener = screen.getByRole("button", { name: "Open" });
    await user.click(opener);

    const dialog = screen.getByRole("dialog", { name: /Figure 2.1/ });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    const close = screen.getByRole("button", { name: "Close enlarged image" });
    expect(close).toHaveFocus();

    await user.tab();
    expect(close).toHaveFocus();
    await user.tab({ shift: true });
    expect(close).toHaveFocus();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it("shows the caption or a neutral fallback", async () => {
    const user = userEvent.setup();
    render(<Harness target={uncaptionedFigure} />);
    await user.click(screen.getByRole("button", { name: "Open" }));
    expect(screen.getByRole("dialog")).toHaveTextContent("No caption");
    await user.click(screen.getByRole("button", { name: "Close enlarged image" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
