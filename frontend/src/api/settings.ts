import { apiRequest, type ApiValue } from "./client";

export async function getMySettings<T>(): Promise<T> {
  return apiRequest<T>("/settings/me");
}

export async function updateMySettings<T>(payload: Record<string, ApiValue>): Promise<T> {
  return apiRequest<T>("/settings/me", { method: "PATCH", body: payload });
}
