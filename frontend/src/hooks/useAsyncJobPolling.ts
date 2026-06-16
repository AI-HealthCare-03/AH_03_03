import { useEffect, useRef, useState } from "react";

import { getAsyncJob, type AsyncJob } from "../api/jobs";

type AsyncJobPollingOptions = {
  jobId: number | null | undefined;
  enabled?: boolean;
  intervalMs?: number;
  timeoutMs?: number;
  onSuccess?: (job: AsyncJob) => void | Promise<void>;
  onFailure?: (job: AsyncJob) => void | Promise<void>;
  onTimeout?: () => void | Promise<void>;
};

const DEFAULT_INTERVAL_MS = 1500;
const DEFAULT_TIMEOUT_MS = 120000;
const TERMINAL_STATUSES = new Set(["SUCCESS", "FAILED", "CANCELED"]);

export function useAsyncJobPolling({
  jobId,
  enabled = true,
  intervalMs = DEFAULT_INTERVAL_MS,
  timeoutMs = DEFAULT_TIMEOUT_MS,
  onSuccess,
  onFailure,
  onTimeout,
}: AsyncJobPollingOptions) {
  const [latestJob, setLatestJob] = useState<AsyncJob | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [pollingError, setPollingError] = useState<Error | null>(null);

  const callbacksRef = useRef({ onSuccess, onFailure, onTimeout });

  useEffect(() => {
    callbacksRef.current = { onSuccess, onFailure, onTimeout };
  }, [onSuccess, onFailure, onTimeout]);

  useEffect(() => {
    if (!enabled || !jobId) {
      setIsPolling(false);
      return undefined;
    }

    let disposed = false;
    let timerId: number | undefined;
    const startedAt = Date.now();

    const stop = () => {
      if (timerId !== undefined) {
        window.clearTimeout(timerId);
      }
      setIsPolling(false);
    };

    const poll = async () => {
      if (disposed) {
        return;
      }
      if (Date.now() - startedAt >= timeoutMs) {
        stop();
        await callbacksRef.current.onTimeout?.();
        return;
      }

      try {
        const job = await getAsyncJob(jobId);
        if (disposed) {
          return;
        }
        setLatestJob(job);
        setPollingError(null);

        if (TERMINAL_STATUSES.has(job.status)) {
          stop();
          try {
            if (job.status === "SUCCESS") {
              await callbacksRef.current.onSuccess?.(job);
              return;
            }
            await callbacksRef.current.onFailure?.(job);
          } catch (err) {
            setPollingError(err instanceof Error ? err : new Error("작업 완료 후 처리를 실패했습니다."));
          }
          return;
        }

        timerId = window.setTimeout(poll, intervalMs);
      } catch (err) {
        if (disposed) {
          return;
        }
        setPollingError(err instanceof Error ? err : new Error("작업 상태를 확인하지 못했습니다."));
        timerId = window.setTimeout(poll, intervalMs);
      }
    };

    setIsPolling(true);
    setPollingError(null);
    setLatestJob(null);
    timerId = window.setTimeout(poll, intervalMs);

    return () => {
      disposed = true;
      if (timerId !== undefined) {
        window.clearTimeout(timerId);
      }
    };
  }, [enabled, intervalMs, jobId, timeoutMs]);

  return { latestJob, isPolling, pollingError };
}
