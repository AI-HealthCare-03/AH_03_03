import { apiRequest, type ApiValue } from "./client";

export type AdminSummary = {
  total_users: number;
  active_users: number;
  today_new_users: number;
  total_health_records: number;
  total_analysis_results: number;
  total_exam_reports: number;
  total_medications: number;
  total_notifications: number;
  system_error_count_today: number;
  sensitive_access_count_today: number;
  email_service_status: string;
  environment: string;
};

export type AdminUsersSummary = {
  total_users: number;
  active_users: number;
  inactive_users: number;
  today_new_users: number;
  monitor_users: number;
  operator_users: number;
  admin_users: number;
  super_admin_users: number;
};

export type AdminSystemHealth = {
  status: string;
  service: string;
  environment: string;
  checks: Record<string, string>;
  details?: Record<string, string>;
};

export type AdminSystemErrorLog = {
  id: number;
  request_id: string | null;
  user_id: number | null;
  method: string;
  path: string;
  status_code: number;
  error_type: string;
  error_message: string | null;
  client_ip: string | null;
  user_agent: string | null;
  created_at: string;
};

export type AdminSensitiveAccessLog = {
  id: number;
  request_id: string | null;
  actor_user_id: number;
  actor_role: string | null;
  target_user_id: number;
  action_type: string;
  resource_type: string;
  resource_id: number | null;
  access_reason: string | null;
  method: string;
  path: string;
  client_ip: string | null;
  user_agent: string | null;
  created_at: string;
};

export type AdminLogList<T> = {
  items: T[];
  total: number;
  limit: number;
  filters: Record<string, string | number | null>;
};

export type AdminFaq = {
  id: number;
  category: string;
  question: string;
  answer: string;
  display_order: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
};

export type AdminFaqPayload = {
  category: string;
  question: string;
  answer: string;
  display_order?: number;
  is_active?: boolean;
};

export type AdminInquiry = {
  id: number;
  user_id: number;
  category: string;
  title: string;
  content: string;
  status: string;
  answer: string | null;
  answered_at: string | null;
  created_at: string;
  updated_at: string;
};

export function getAdminSummary(): Promise<AdminSummary> {
  return apiRequest<AdminSummary>("/admin/summary");
}

export function getAdminUsersSummary(): Promise<AdminUsersSummary> {
  return apiRequest<AdminUsersSummary>("/admin/users/summary");
}

export function getAdminSystemHealth(): Promise<AdminSystemHealth> {
  return apiRequest<AdminSystemHealth>("/admin/system/health");
}

export function getAdminSystemErrors(limit = 50): Promise<AdminLogList<AdminSystemErrorLog>> {
  return apiRequest<AdminLogList<AdminSystemErrorLog>>(`/admin/system/errors?limit=${limit}`);
}

export function getAdminSensitiveAccessLogs(limit = 50): Promise<AdminLogList<AdminSensitiveAccessLog>> {
  return apiRequest<AdminLogList<AdminSensitiveAccessLog>>(`/admin/sensitive-access-logs?limit=${limit}`);
}

export function listAdminFaqs(params: { category?: string; isActive?: boolean } = {}): Promise<AdminFaq[]> {
  const query = new URLSearchParams();
  query.set("limit", "100");
  if (params.category) query.set("category", params.category);
  if (params.isActive !== undefined) query.set("is_active", String(params.isActive));
  return apiRequest<AdminFaq[]>(`/admin/faqs?${query.toString()}`);
}

export function createAdminFaq(payload: AdminFaqPayload): Promise<AdminFaq> {
  return apiRequest<AdminFaq>("/admin/faqs", { method: "POST", body: payload as Record<string, ApiValue> });
}

export function updateAdminFaq(faqId: number, payload: Partial<AdminFaqPayload>): Promise<AdminFaq> {
  return apiRequest<AdminFaq>(`/admin/faqs/${faqId}`, {
    method: "PATCH",
    body: payload as Record<string, ApiValue>,
  });
}

export function deactivateAdminFaq(faqId: number): Promise<AdminFaq> {
  return apiRequest<AdminFaq>(`/admin/faqs/${faqId}`, { method: "DELETE" });
}

export function listAdminInquiries(params: { status?: string; category?: string } = {}): Promise<AdminInquiry[]> {
  const query = new URLSearchParams();
  query.set("limit", "100");
  if (params.status) query.set("status", params.status);
  if (params.category) query.set("category", params.category);
  return apiRequest<AdminInquiry[]>(`/admin/inquiries?${query.toString()}`);
}

export function getAdminInquiry(inquiryId: number): Promise<AdminInquiry> {
  return apiRequest<AdminInquiry>(`/admin/inquiries/${inquiryId}`);
}

export function answerAdminInquiry(inquiryId: number, answer: string): Promise<AdminInquiry> {
  return apiRequest<AdminInquiry>(`/admin/inquiries/${inquiryId}/answer`, {
    method: "POST",
    body: { answer },
  });
}
