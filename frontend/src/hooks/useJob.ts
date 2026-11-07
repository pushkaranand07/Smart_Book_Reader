import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getJob } from "../api/client";
import { queryKeys } from "../api/queryKeys";
import type { Job } from "../api/types";

export const DEFAULT_POLL_MS = 1000;

export function isTerminal(job: Job | undefined): boolean {
  return job?.status === "succeeded" || job?.status === "failed";
}

/** Poll `GET /api/jobs/{job_id}` until the job succeeds or fails. */
export function useJob(jobId: string | null, intervalMs = DEFAULT_POLL_MS) {
  const queryClient = useQueryClient();
  return useQuery({
    queryKey: queryKeys.job(jobId ?? ""),
    queryFn: async ({ signal }) => {
      const job = await getJob(jobId ?? "", signal);
      if (job.status === "succeeded") void queryClient.invalidateQueries({ queryKey: queryKeys.books, exact: true });
      return job;
    },
    enabled: jobId !== null,
    staleTime: 0,
    refetchInterval: (query) => (isTerminal(query.state.data) ? false : intervalMs),
    refetchIntervalInBackground: true,
  });
}

export function jobErrorMessage(job: Job): string {
  if (typeof job.error === "string" && job.error) return job.error;
  if (job.error && typeof job.error === "object") return job.error.message;
  return job.message ?? "Processing failed.";
}
