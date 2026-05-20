import { apiRequest, type ApiValue } from "./client";

export async function listDietRecords<T>(): Promise<T> {
  return apiRequest<T>("/diets");
}

export async function createDietRecord<T>(payload: Record<string, ApiValue>): Promise<T> {
  return apiRequest<T>("/diets", { method: "POST", body: payload });
}

export async function runDummyDietAnalysis<T>(payload: Record<string, ApiValue>): Promise<T> {
  return apiRequest<T>("/diets/dummy-analyze", { method: "POST", body: payload });
}
