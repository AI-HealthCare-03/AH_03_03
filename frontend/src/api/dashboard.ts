import { apiRequest } from "./client";

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
