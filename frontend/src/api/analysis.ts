import { apiRequest } from "./client";
import type { AsyncJob } from "./jobs";

export type AnalysisMode = "BASIC" | "PRECISION";

export type AnalysisRunResponse = {
  analysis_result_id: number;
  analysis_type: string;
  analysis_mode: AnalysisMode;
  risk_score: string | number;
  risk_level: string;
  model_name?: string | null;
  model_version?: string | null;
  guide_message: string;
  explanation?: {
    summary?: string;
    caution?: string;
    recommended_action?: string;
    safety_notice?: string;
    source?: string;
    reference_summary?: string | null;
    reference_sources?: Array<{
      id?: string | null;
      title?: string | null;
      source_org?: string | null;
      source_url?: string | null;
      year?: number | null;
      status?: string | null;
    }>;
  } | null;
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

export async function runAnalysisAsync(
  healthRecordId: number,
  mode: AnalysisMode = "BASIC",
): Promise<AsyncJob> {
  return apiRequest<AsyncJob>("/analysis/run-async", {
    method: "POST",
    body: { health_record_id: healthRecordId, mode },
  });
}
