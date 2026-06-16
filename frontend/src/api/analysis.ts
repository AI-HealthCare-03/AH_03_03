import { apiRequest } from "./client";
import type { AsyncJob } from "./jobs";

export type AnalysisMode = "BASIC" | "PRECISION";

export type AnalysisRunResponse = {
  analysis_result_id: number;
  analysis_type: string;
  analysis_mode: AnalysisMode;
  risk_score: string | number;
  risk_level: string;
  service_band?: string | null;
  service_band_label?: string | null;
  service_band_percent?: number | null;
  legacy_risk_level?: string | null;
  result_source?: string | null;
  x2_stage_code?: string | null;
  x2_stage_label?: string | null;
  x2_available?: boolean | null;
  x2_missing_fields?: string[] | null;
  selected_exam_report_id?: number | null;
  x2_measurement_source?: string | null;
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

export type AnalysisResultResponse = {
  id: number;
  analysis_type: string;
  analysis_mode: AnalysisMode;
  analyzed_at?: string | null;
  created_at?: string | null;
  async_job_id?: number | null;
  risk_score?: string | number | null;
  risk_level?: string | null;
  service_band?: string | null;
  service_band_label?: string | null;
  service_band_percent?: number | null;
  legacy_risk_level?: string | null;
  result_source?: string | null;
  x2_stage_code?: string | null;
  x2_stage_label?: string | null;
  x2_available?: boolean | null;
  x2_missing_fields?: string[] | null;
  selected_exam_report_id?: number | null;
  x2_measurement_source?: string | null;
  summary?: string | null;
  guide_message?: string | null;
  message?: string | null;
};

export async function getLatestAnalysisResults<T>(): Promise<T> {
  return apiRequest<T>("/analysis/results/latest");
}

export async function listAnalysisResults<T>(limit = 100): Promise<T> {
  return apiRequest<T>(`/analysis/results?limit=${limit}`);
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
