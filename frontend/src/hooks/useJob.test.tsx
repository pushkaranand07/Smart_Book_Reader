import { waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { queryKeys } from "../api/queryKeys";
import { errorBody, job, jsonResponse, JOB_ID } from "../test/fixtures";
import { renderHookWithClient } from "../test/utils";
import { jobErrorMessage, useJob } from "./useJob";

describe("useJob", () => {
  it("polls until the job succeeds, then stops and refreshes the book list", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(job({ status: "queued", progress: 0, message: "Queued" })))
      .mockResolvedValueOnce(jsonResponse(job({ status: "running", progress: 0.5 })))
      .mockResolvedValueOnce(jsonResponse(job({ status: "succeeded", progress: 1, message: "Done" })));
    vi.stubGlobal("fetch", fetchMock);

    const { result, client } = renderHookWithClient(() => useJob(JOB_ID, 10));
    const invalidate = vi.spyOn(client, "invalidateQueries");

    await waitFor(() => {
      expect(result.current.data?.status).toBe("succeeded");
    });
    expect(result.current.data?.progress).toBe(1);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(`/api/jobs/${JOB_ID}`);
    expect(invalidate).toHaveBeenCalledWith({ queryKey: queryKeys.books, exact: true });

    // No further polling once terminal.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("stops polling on failure and exposes the job error", async () => {
    const failed = job({ status: "failed", progress: 0.3, error: { code: "PDF_CORRUPT", message: "Corrupt PDF" } });
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(job()))
      .mockResolvedValueOnce(jsonResponse(failed));
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHookWithClient(() => useJob(JOB_ID, 10));
    await waitFor(() => {
      expect(result.current.data?.status).toBe("failed");
    });
    expect(jobErrorMessage(result.current.data ?? failed)).toBe("Corrupt PDF");
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not fetch without a job id", () => {
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", fetchMock);
    const { result } = renderHookWithClient(() => useJob(null, 10));
    expect(result.current.fetchStatus).toBe("idle");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("surfaces a typed API error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>(() => Promise.resolve(jsonResponse(errorBody("JOB_NOT_FOUND", "Unknown job"), 404))),
    );
    const { result } = renderHookWithClient(() => useJob(JOB_ID, 10));
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });
    expect(result.current.error).toMatchObject({ code: "JOB_NOT_FOUND" });
  });

  it("formats string errors and falls back to the message", () => {
    expect(jobErrorMessage(job({ status: "failed", error: "boom" }))).toBe("boom");
    expect(jobErrorMessage(job({ status: "failed", error: null, message: "Stopped" }))).toBe("Stopped");
  });
});
