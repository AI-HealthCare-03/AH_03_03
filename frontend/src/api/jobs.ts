import { apiRequest } from "./client";

export type AsyncJobStatus = "PENDING" | "PROCESSING" | "SUCCESS" | "FAILED" | "CANCELED";

export type AsyncJob = {
  id: number;
  job_type: string;
  status: AsyncJobStatus;
  request_payload?: Record<string, unknown> | null;
  result_payload?: Record<string, unknown> | null;
  error_message?: string | null;
  stream_id?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  finished_at?: string | null;
};

export async function getAsyncJob(jobId: number): Promise<AsyncJob> {
  return apiRequest<AsyncJob>(`/jobs/${jobId}`);
}
