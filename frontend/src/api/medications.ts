import { apiRequest, type ApiValue } from "./client";

export async function listMedications<T>(): Promise<T> {
  return apiRequest<T>("/medications");
}

export async function createMedication<T>(payload: Record<string, ApiValue>): Promise<T> {
  return apiRequest<T>("/medications", { method: "POST", body: payload });
}

export async function listMedicationRecords<T>(medicationId: number): Promise<T> {
  return apiRequest<T>(`/medications/${medicationId}/records`);
}

export async function updateMedicationRecord<T>(recordId: number, payload: Record<string, ApiValue>): Promise<T> {
  return apiRequest<T>(`/medications/records/${recordId}`, { method: "PATCH", body: payload });
}
