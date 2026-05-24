import { apiRequest, type ApiValue } from "./client";

export type MedicationOcrItem = {
  temp_id?: string | null;
  name: string;
  dosage?: string | null;
  frequency?: string | null;
  time_slots: string[];
  duration_days?: number | null;
  memo?: string | null;
  confidence?: number | null;
};

export type MedicationOcrRequest = {
  source_type?: "PRESCRIPTION" | "MEDICATION_BAG" | "SUPPLEMENT" | string;
  image_filename?: string;
  memo?: string;
  raw_text?: string;
};

export type MedicationOcrResponse = {
  is_dummy: boolean;
  source_type: string;
  ocr_confidence: number;
  items: MedicationOcrItem[];
  message: string;
  source?: string;
  raw_text?: string | null;
  parser_warnings?: string[];
};

export type MedicationOcrConfirmRequest = {
  items: Array<{
    name: string;
    dosage?: string | null;
    frequency?: string | null;
    time_slots: string[];
    duration_days?: number | null;
    memo?: string | null;
  }>;
};

export type MedicationOcrConfirmResponse = {
  created_count: number;
  created_medication_ids: number[];
  skipped_count: number;
  message: string;
};

export type MedicationPayload = Record<string, ApiValue> & {
  name?: string;
  medication_type?: string;
  dosage?: string | null;
  frequency?: string | null;
  reminder_time?: string | null;
  is_active?: boolean;
  memo?: string | null;
};

export async function listMedications<T>(): Promise<T> {
  return apiRequest<T>("/medications");
}

export async function getMedication<T>(medicationId: number): Promise<T> {
  return apiRequest<T>(`/medications/${medicationId}`);
}

export async function createMedication<T>(payload: MedicationPayload): Promise<T> {
  return apiRequest<T>("/medications", { method: "POST", body: payload });
}

export async function updateMedication<T>(medicationId: number, payload: Partial<MedicationPayload>): Promise<T> {
  return apiRequest<T>(`/medications/${medicationId}`, { method: "PATCH", body: payload });
}

export async function deactivateMedication<T>(medicationId: number): Promise<T> {
  return apiRequest<T>(`/medications/${medicationId}/deactivate`, { method: "PATCH" });
}

export async function deleteMedication<T>(medicationId: number): Promise<T> {
  return apiRequest<T>(`/medications/${medicationId}`, { method: "DELETE" });
}

export async function runMedicationOcr(
  payload: MedicationOcrRequest,
): Promise<MedicationOcrResponse> {
  return apiRequest<MedicationOcrResponse>("/medications/ocr", { method: "POST", body: payload });
}

export async function confirmMedicationOcr(
  payload: MedicationOcrConfirmRequest,
): Promise<MedicationOcrConfirmResponse> {
  return apiRequest<MedicationOcrConfirmResponse>("/medications/ocr-confirm", { method: "POST", body: payload });
}

export async function listMedicationRecords<T>(medicationId: number): Promise<T> {
  return apiRequest<T>(`/medications/${medicationId}/records`);
}

export async function updateMedicationRecord<T>(recordId: number, payload: Record<string, ApiValue>): Promise<T> {
  return apiRequest<T>(`/medications/records/${recordId}`, { method: "PATCH", body: payload });
}
