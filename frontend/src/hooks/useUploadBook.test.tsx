import { act, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { BOOK_ID, JOB_ID, errorBody } from "../test/fixtures";
import { MockXhr } from "../test/mockXhr";
import { renderHookWithClient } from "../test/utils";
import { useUploadBook } from "./useUploadBook";

describe("useUploadBook", () => {
  beforeEach(() => {
    MockXhr.reset();
    vi.stubGlobal("XMLHttpRequest", MockXhr);
  });

  it("exposes upload progress and resolves with job ids", async () => {
    const { result } = renderHookWithClient(() => useUploadBook());
    const file = new File(["%PDF"], "book.pdf", { type: "application/pdf" });

    act(() => {
      result.current.mutate(file);
    });
    await waitFor(() => {
      expect(result.current.isPending).toBe(true);
    });
    expect(result.current.progress).toBe(0);

    act(() => {
      MockXhr.latest().progress(40, 100);
    });
    expect(result.current.progress).toBe(0.4);

    act(() => {
      MockXhr.latest().respond(202, { book_id: BOOK_ID, job_id: JOB_ID, status: "queued" });
    });
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });
    expect(result.current.data).toEqual({ book_id: BOOK_ID, job_id: JOB_ID, status: "queued" });
    expect(result.current.progress).toBe(1);
  });

  it("exposes typed errors", async () => {
    const { result } = renderHookWithClient(() => useUploadBook());
    act(() => {
      result.current.mutate(new File(["x"], "huge.pdf"));
    });
    await waitFor(() => {
      expect(MockXhr.instances).toHaveLength(1);
    });
    act(() => {
      MockXhr.latest().respond(413, errorBody("FILE_TOO_LARGE", "Max 50 MB"));
    });
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });
    expect(result.current.error).toMatchObject({ status: 413, code: "FILE_TOO_LARGE", message: "Max 50 MB" });
  });
});
