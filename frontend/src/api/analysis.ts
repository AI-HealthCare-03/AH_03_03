import { apiRequest } from "./client";

export async function getLatestAnalysisResults<T>(): Promise<T> {
  return apiRequest<T>("/analysis/results/latest");
}

export async function listAnalysisResults<T>(): Promise<T> {
  return apiRequest<T>("/analysis/results");
}

export async function getAnalysisResult<T>(analysisId: number): Promise<T> {
  return apiRequest<T>(`/analysis/results/${analysisId}`);
}

export async function getAnalysisResultDetail<T>(analysisId: number): Promise<T> {
  return apiRequest<T>(`/analysis/results/${analysisId}/detail`);
}

export async function runDummyAnalysis<T>(healthRecordId: number): Promise<T> {
  return apiRequest<T>("/analysis/dummy-run", {
    method: "POST",
    body: { health_record_id: healthRecordId },
  });
}
