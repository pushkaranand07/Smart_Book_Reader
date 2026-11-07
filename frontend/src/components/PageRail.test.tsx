import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { bookDetail } from "../test/fixtures";
import { PageRail } from "./PageRail";

function Harness() {
  const [selected, setSelected] = useState<number | null>(1);
  return <PageRail pages={bookDetail.pages} selectedPage={selected} onSelect={setSelected} />;
}

describe("PageRail", () => {
  it("is a single tab stop with arrow / Home / End navigation", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(3);
    expect(options[0]).toHaveAttribute("aria-selected", "true");
    expect(options.filter((option) => option.tabIndex === 0)).toHaveLength(1);

    await user.tab();
    expect(options[0]).toHaveFocus();

    await user.keyboard("{ArrowDown}");
    expect(screen.getByRole("option", { name: /Page 2/ })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("option", { name: /Page 2/ })).toHaveFocus();

    await user.keyboard("{End}");
    expect(screen.getByRole("option", { name: /Page 3/ })).toHaveFocus();
    await user.keyboard("{ArrowDown}");
    expect(screen.getByRole("option", { name: /Page 3/ })).toHaveAttribute("aria-selected", "true");

    await user.keyboard("{Home}");
    expect(screen.getByRole("option", { name: /Page 1/ })).toHaveFocus();
    await user.keyboard("{ArrowUp}");
    expect(screen.getByRole("option", { name: /Page 1/ })).toHaveAttribute("aria-selected", "true");
  });

  it("selects on click", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole("option", { name: /Page 3/ }));
    expect(screen.getByRole("option", { name: /Page 3/ })).toHaveAttribute("aria-selected", "true");
  });
});
