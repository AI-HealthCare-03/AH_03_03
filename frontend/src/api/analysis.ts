import { apiRequest } from "./client";

export type AnalysisMode = "BASIC" | "PRECISION";

export type AnalysisRunResponse = {
  analysis_result_id: number;
  analysis_type: string;
  analysis_mode: AnalysisMode;
  risk_score: string | number;
  risk_level: string;
  guide_message: string;
  challenge_recommendation_ids: number[];
  factor_count: number;
};

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

export async function runDummyAnalysis<T = AnalysisRunResponse[]>(
  healthRecordId: number,
  mode: AnalysisMode = "BASIC",
): Promise<T> {
  return apiRequest<T>("/analysis/dummy-run", {
    method: "POST",
    body: { health_record_id: healthRecordId, mode },
  });
}
