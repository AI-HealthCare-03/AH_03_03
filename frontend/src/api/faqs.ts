import { apiRequest, type ApiValue } from "./client";

export async function listFaqs<T>(): Promise<T> {
  return apiRequest<T>("/faqs", { skipAuth: true });
}

export async function createInquiry<T>(payload: Record<string, ApiValue>): Promise<T> {
  return apiRequest<T>("/inquiries", { method: "POST", body: payload });
}

export async function listMyInquiries<T>(): Promise<T> {
  return apiRequest<T>("/inquiries/my");
}
