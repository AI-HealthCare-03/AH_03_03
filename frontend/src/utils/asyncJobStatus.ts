import type { AsyncJobStatus } from "../api/jobs";

export type AsyncJobUiStatus = AsyncJobStatus | "RUNNING" | "COMPLETED" | "ERROR" | "CANCELLED" | "TIMEOUT";

export const ASYNC_JOB_STATUS_MESSAGES: Record<AsyncJobUiStatus, string> = {
  PENDING: "분석 대기 중",
  PROCESSING: "분석 중",
  RUNNING: "분석 중",
  SUCCESS: "분석 완료",
  COMPLETED: "분석 완료",
  FAILED: "분석 실패",
  ERROR: "분석 실패",
  CANCELED: "취소됨",
  CANCELLED: "취소됨",
  TIMEOUT: "분석 시간이 길어지고 있습니다",
};

export function normalizeAsyncJobStatus(status: unknown): string {
  return String(status ?? "").trim().toUpperCase();
}

export function getAsyncJobStatusLabel(status: unknown): string {
  return ASYNC_JOB_STATUS_MESSAGES[normalizeAsyncJobStatus(status) as AsyncJobUiStatus] ?? "분석 상태 확인 중";
}

export function getAsyncJobStatusMessage(status: unknown): string {
  return getAsyncJobStatusLabel(status);
}
