import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { MOCK_API_KEY, mockApi } from "./mockApi";

const realApi = process.env.E2E_REAL_API === "1";
const apiKey = realApi ? (process.env.E2E_API_KEY ?? "") : MOCK_API_KEY;
const question = process.env.E2E_QUESTION ?? "Where do the light-dependent reactions take place?";

function pdfFile() {
  if (realApi) {
    const pdfPath = process.env.E2E_PDF;
    if (!pdfPath) throw new Error("E2E_REAL_API=1 needs E2E_PDF (path to a PDF), E2E_API_URL and E2E_API_KEY");
    return { name: path.basename(pdfPath), mimeType: "application/pdf", buffer: readFileSync(pdfPath) };
  }
  const minimalPdf = "%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n";
  return { name: "photosynthesis.pdf", mimeType: "application/pdf", buffer: Buffer.from(minimalPdf) };
}

/** Fail on serious/critical axe violations (WCAG 2.1 A/AA); print a summary of everything found. */
async function expectNoSeriousA11yViolations(page: Page, label: string) {
  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  const summary = results.violations.map((violation) => `${violation.impact ?? "?"}: ${violation.id}`);
  console.log(`[axe] ${label}: ${results.violations.length} violation(s) ${JSON.stringify(summary)}`);
  const blocking = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(blocking, `serious/critical axe violations on ${label}`).toEqual([]);
}

test("upload a PDF, wait for the job, ask, and jump to the cited page", async ({ page }) => {
  const mock = realApi ? null : await mockApi(page);
  await page.goto("/");

  await expect(page.getByRole("heading", { name: /Read the page/ })).toBeVisible();
  await expect(page.getByText("API connected")).toBeVisible();
  await expectNoSeriousA11yViolations(page, "empty state");

  // API key: entered through the settings popover (mock mode first proves the 401 prompt appears).
  if (!realApi) await expect(page.getByRole("dialog", { name: "API access" })).toBeVisible();
  if (apiKey) {
    const dialog = page.getByRole("dialog", { name: "API access" });
    if (!(await dialog.isVisible())) await page.getByRole("button", { name: /API key/ }).click();
    await dialog.getByLabel("API key").fill(apiKey);
    await dialog.getByRole("button", { name: "Save key" }).click();
    await expect(dialog).toBeHidden();
  }

  // Upload -> job polling -> book detail.
  await page.getByLabel(/Choose file/).setInputFiles(pdfFile());
  const rail = page.getByRole("listbox", { name: "Pages" });
  await expect(rail).toBeVisible({ timeout: realApi ? 10 * 60_000 : 15_000 });
  if (mock) expect(mock.jobPolls).toBeGreaterThanOrEqual(3);
  await expect(page.getByTestId("status-announcer")).toHaveText(/is ready/);

  // Ask a question and wait for a cited answer.
  await page.getByLabel("Your question about this book").fill(question);
  await page.getByRole("button", { name: /^Ask/ }).click();
  const answer = page.getByTestId("answer");
  await expect(answer).toBeVisible();
  await expect(answer.getByText(/^Mode: (Offline|Gemini)$/)).toBeVisible();
  if (mock) {
    expect(mock.questions).toEqual([question]);
    await expect(answer.locator(".katex").first()).toBeVisible();
    await expect(answer.getByText("Exact passage from the book")).toBeVisible();
  }

  const citation = answer.locator(".citation__button").first();
  await expect(citation).toBeVisible();
  const citedPage = Number(await citation.getAttribute("data-page-number"));
  expect(citedPage).toBeGreaterThan(0);

  // Move away from the cited page first so the jump is observable.
  const firstOption = rail.getByRole("option").first();
  await firstOption.click();
  await citation.click();
  await expect(rail.getByRole("option", { name: new RegExp(`Page ${citedPage},`) })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.getByTestId("page-view")).toHaveAttribute("data-page-number", String(citedPage));
  await expect(page.locator("#page-view-heading")).toBeFocused();

  await expectNoSeriousA11yViolations(page, "book with answer");

  // Keyboard: arrow keys move through the page rail.
  if (mock) {
    await rail.getByRole("option", { name: /Page 1,/ }).click();
    await page.keyboard.press("ArrowDown");
    await expect(rail.getByRole("option", { name: /Page 2,/ })).toBeFocused();
    await expect(page.getByTestId("page-view")).toHaveAttribute("data-page-number", "2");

    // Lightbox: opens, traps focus, closes on Escape and returns focus.
    const opener = page.getByTestId("page-view").getByRole("button", { name: "Enlarge Figure 2.1" });
    await opener.click();
    const dialog = page.getByRole("dialog", { name: /Figure 2.1/ });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole("button", { name: "Close enlarged image" })).toBeFocused();
    await expectNoSeriousA11yViolations(page, "lightbox open");
    await page.keyboard.press("Escape");
    await expect(dialog).toBeHidden();
    await expect(opener).toBeFocused();
  }
});
