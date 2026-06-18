import { apiRequest, type ApiValue } from "./client";

export type HealthRecordPayload = Record<string, ApiValue> & {
  height_cm?: number;
  weight_kg?: number;
  bmi?: number;
  waist_cm?: number;
  systolic_bp?: number;
  diastolic_bp?: number;
  fasting_glucose?: number;
  hba1c?: number;
  total_cholesterol?: number;
  ldl_cholesterol?: number;
  hdl_cholesterol?: number;
  triglyceride?: number;
  ast?: number;
  alt?: number;
  gamma_gtp?: number;
  creatinine?: number;
  egfr?: number;
  hemoglobin?: number;
  has_diabetes?: boolean;
  has_obesity?: boolean;
  has_dyslipidemia?: boolean;
  has_hypertension?: boolean;
  occupation_code?: string;
  family_htn?: string;
  family_dm?: string;
  family_dyslipidemia?: string;
  smoking_status?: string;
  drinking_frequency?: string;
  drinking_amount?: string;
  walking_days_per_week?: number;
  strength_days_per_week?: number;
  sleep_hours?: number;
  source?: "MANUAL" | "OCR" | "PROFILE" | "ANALYSIS_PREP";
  measured_at: string;
};

export async function listHealthRecords<T>(): Promise<T> {
  return apiRequest<T>("/health/records");
}

export async function getLatestHealthRecord<T>(): Promise<T> {
  return apiRequest<T>("/health/records/latest");
}

export async function createHealthRecord<T>(payload: HealthRecordPayload): Promise<T> {
  return apiRequest<T>("/health/records", { method: "POST", body: payload });
}

export async function updateHealthRecord<T>(
  recordId: number,
  payload: Partial<HealthRecordPayload>,
): Promise<T> {
  return apiRequest<T>(`/health/records/${recordId}`, { method: "PATCH", body: payload });
}

export async function deleteHealthRecord<T>(recordId: number): Promise<T> {
  return apiRequest<T>(`/health/records/${recordId}`, { method: "DELETE" });
}

export async function getAnalysisReadiness<T>(): Promise<T> {
  return apiRequest<T>("/health/analysis-readiness");
}
