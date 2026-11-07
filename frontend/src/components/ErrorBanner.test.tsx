import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import { ErrorBanner } from "./ErrorBanner";

describe("ErrorBanner", () => {
  it("renders code, message and request id of a typed API error", () => {
    render(<ErrorBanner error={new ApiError(404, "BOOK_NOT_FOUND", "No such book", "req-9")} title="Oops" />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Oops");
    expect(alert).toHaveTextContent("No such book");
    expect(alert).toHaveTextContent("Code: BOOK_NOT_FOUND");
    expect(alert).toHaveTextContent("Request ID: req-9");
  });

  it("handles plain errors without a request id", () => {
    render(<ErrorBanner error={new Error("Broken")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Broken");
    expect(screen.queryByText(/Request ID/)).not.toBeInTheDocument();
  });

  it("wires retry and dismiss actions", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    const onDismiss = vi.fn();
    render(<ErrorBanner error={new Error("x")} onRetry={onRetry} onDismiss={onDismiss} />);
    await user.click(screen.getByRole("button", { name: "Try again" }));
    await user.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(onRetry).toHaveBeenCalledOnce();
    expect(onDismiss).toHaveBeenCalledOnce();
  });
});
