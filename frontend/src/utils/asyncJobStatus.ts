import type { AsyncJobStatus } from "../api/jobs";

export type AsyncJobUiStatus = AsyncJobStatus | "TIMEOUT";

export const ASYNC_JOB_STATUS_MESSAGES: Record<AsyncJobUiStatus, string> = {
  PENDING: "작업 대기 중입니다.",
  PROCESSING: "분석 중입니다.",
  SUCCESS: "분석이 완료되었습니다.",
  FAILED: "분석에 실패했습니다.",
  CANCELED: "작업이 취소되었습니다.",
  TIMEOUT: "분석 시간이 길어지고 있습니다.",
};

export function getAsyncJobStatusMessage(status: AsyncJobUiStatus): string {
  return ASYNC_JOB_STATUS_MESSAGES[status];
}
