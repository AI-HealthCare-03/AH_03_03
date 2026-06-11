import { apiRequest } from "./client";

export type DashboardAnalysisResult = {
  id: number;
  analysis_type: string;
  analysis_mode?: "BASIC" | "PRECISION";
  risk_level: string;
  risk_score: number;
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
  model_name?: string | null;
  model_version?: string | null;
  analyzed_at: string;
  created_at: string;
};

export type DashboardRiskFactor = {
  analysis_result_id: number;
  analysis_type: string;
  analysis_mode?: "BASIC" | "PRECISION";
  factor_key: string;
  factor_name: string;
  factor_value?: string | null;
  contribution_score?: number | null;
  direction: string;
};

export type DashboardSummary = {
  latest_health_record?: Record<string, unknown> | null;
  unread_notification_count: number;
  active_challenge_count: number;
  active_medication_count: number;
  latest_analysis_results: DashboardAnalysisResult[];
  top_risk_factors: DashboardRiskFactor[];
  overall_risk_level?: string | null;
  overall_risk_score?: number | null;
};

export type DashboardRiskTrendPoint = {
  analyzed_at: string;
  created_at?: string | null;
  id?: number | string | null;
  risk_score: number;
  risk_level: string;
  service_band?: string | null;
  service_band_label?: string | null;
  service_band_percent?: number | null;
  legacy_risk_level?: string | null;
};

export type DashboardRiskTrendSeries = {
  disease_type: string;
  points: DashboardRiskTrendPoint[];
};

export type DashboardRiskTrend = {
  period: string;
  date_from?: string | null;
  date_to?: string | null;
  series: DashboardRiskTrendSeries[];
};

export async function getDashboardSummary<T>(): Promise<T> {
  return apiRequest<T>("/dashboard/summary");
}

export async function getDashboardHealth<T>(): Promise<T> {
  return apiRequest<T>("/dashboard/health");
}

export async function getDashboardChallenges<T>(): Promise<T> {
  return apiRequest<T>("/dashboard/challenges");
}

export async function getDashboardDiets<T>(): Promise<T> {
  return apiRequest<T>("/dashboard/diets");
}

export async function getDashboardMedications<T>(): Promise<T> {
  return apiRequest<T>("/dashboard/medications");
}

export async function getDashboardTrends<T>(period = "week"): Promise<T> {
  return apiRequest<T>(`/dashboard/trends?period=${encodeURIComponent(period)}`);
}

export async function getDashboardRiskTrend<T>(period = "all"): Promise<T> {
  return apiRequest<T>(`/dashboard/risk-trend?period=${encodeURIComponent(period)}`);
}
