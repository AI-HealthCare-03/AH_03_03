import { apiRequest, type ApiValue } from "./client";

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

export async function listMedicationRecords<T>(medicationId: number): Promise<T> {
  return apiRequest<T>(`/medications/${medicationId}/records`);
}

export async function updateMedicationRecord<T>(recordId: number, payload: Record<string, ApiValue>): Promise<T> {
  return apiRequest<T>(`/medications/records/${recordId}`, { method: "PATCH", body: payload });
}
