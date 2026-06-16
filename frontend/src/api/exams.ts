import { apiRequest } from "./client";
import type { AsyncJob } from "./jobs";

export type ExamReport = {
  id: number;
  user_id: number;
  original_filename: string;
  file_path: string;
  exam_date?: string | null;
  ocr_status: string;
  is_confirmed: boolean;
  uploaded_at: string;
  confirmed_at?: string | null;
  created_at: string;
  updated_at: string;
};

export type ExamMeasurement = {
  id: number;
  exam_report_id: number;
  measurement_key: string;
  measurement_name: string;
  value?: string | null;
  unit?: string | null;
  ocr_confidence?: string | number | null;
  is_user_confirmed: boolean;
  created_at: string;
  updated_at: string;
};

export type ExamOcrResponse = {
  message: string;
  measurements: ExamMeasurement[];
  ocr_provider?: string;
  fallback_used?: boolean;
  provider_message?: string | null;
  raw_text_preview?: string | null;
};

export async function listExams<T = ExamReport[]>(params?: { limit?: number; offset?: number }): Promise<T> {
  const query = new URLSearchParams();
  if (params?.limit != null) query.set("limit", String(params.limit));
  if (params?.offset != null) query.set("offset", String(params.offset));
  const qs = query.toString();
  return apiRequest<T>(`/exams${qs ? `?${qs}` : ""}`);
}

export async function getExam(examId: number): Promise<ExamReport> {
  return apiRequest<ExamReport>(`/exams/${examId}`);
}

export async function createExam(payload: {
  original_filename: string;
  file_path: string;
  exam_date?: string | null;
  uploaded_at: string;
}): Promise<ExamReport> {
  return apiRequest<ExamReport>("/exams", { method: "POST", body: payload });
}

export async function runExamOcr(examId: number, file: File): Promise<AsyncJob> {
  const formData = new FormData();
  formData.append("image", file);
  return apiRequest<AsyncJob>(`/exams/${examId}/ocr`, { method: "POST", body: formData });
}

export async function listMeasurements(examId: number): Promise<ExamMeasurement[]> {
  return apiRequest<ExamMeasurement[]>(`/exams/${examId}/measurements`);
}

export async function updateMeasurement(
  measurementId: number,
  payload: Partial<Pick<ExamMeasurement, "measurement_key" | "measurement_name" | "value" | "unit" | "ocr_confidence" | "is_user_confirmed">>,
): Promise<ExamMeasurement> {
  return apiRequest<ExamMeasurement>(`/exams/measurements/${measurementId}`, { method: "PATCH", body: payload });
}

export async function confirmExam(examId: number): Promise<ExamReport> {
  return apiRequest<ExamReport>(`/exams/${examId}/confirm`, { method: "POST" });
}
