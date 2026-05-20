import { apiRequest, type ApiValue } from "./client";

export async function listHealthRecords<T>(): Promise<T> {
  return apiRequest<T>("/health/records");
}

export async function createHealthRecord<T>(payload: Record<string, ApiValue>): Promise<T> {
  return apiRequest<T>("/health/records", { method: "POST", body: payload });
}

export async function getAnalysisReadiness<T>(): Promise<T> {
  return apiRequest<T>("/health/analysis-readiness");
}
