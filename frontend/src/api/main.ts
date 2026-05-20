import { apiRequest } from "./client";

export async function getPublicMain<T>(): Promise<T> {
  return apiRequest<T>("/main/public", { skipAuth: true });
}

export async function getMainSummary<T>(): Promise<T> {
  return apiRequest<T>("/main/summary");
}
