import { useQuery } from "@tanstack/react-query";
import { checkHealth } from "../api/client";
import { queryKeys } from "../api/queryKeys";

export type HealthStatus = "checking" | "connected" | "unavailable";

export function useHealth(intervalMs = 30_000): HealthStatus {
  const query = useQuery({
    queryKey: queryKeys.health,
    queryFn: ({ signal }) => checkHealth(signal),
    retry: false,
    refetchInterval: intervalMs,
  });
  if (query.isPending) return "checking";
  return query.isSuccess ? "connected" : "unavailable";
}
