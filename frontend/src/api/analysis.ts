import { apiRequest } from "./client";

export async function getLatestAnalysisResults<T>(): Promise<T> {
  return apiRequest<T>("/analysis/results/latest");
}

export async function runDummyAnalysis<T>(healthRecordId: number): Promise<T> {
  return apiRequest<T>("/analysis/dummy-run", {
    method: "POST",
    body: { health_record_id: healthRecordId },
  });
}
