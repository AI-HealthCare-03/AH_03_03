import { apiRequest } from "./client";

export async function getDashboardSummary<T>(): Promise<T> {
  return apiRequest<T>("/dashboard/summary");
}

export async function getDashboardTrends<T>(period = "week"): Promise<T> {
  return apiRequest<T>(`/dashboard/trends?period=${encodeURIComponent(period)}`);
}
