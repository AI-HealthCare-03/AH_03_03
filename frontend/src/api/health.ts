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
  has_diabetes?: boolean;
  has_obesity?: boolean;
  has_dyslipidemia?: boolean;
  has_hypertension?: boolean;
  is_smoker?: boolean;
  drinks_alcohol?: boolean;
  exercise_days_per_week?: number;
  sleep_hours?: number;
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

export async function getAnalysisReadiness<T>(): Promise<T> {
  return apiRequest<T>("/health/analysis-readiness");
}
