import { apiRequest, type ApiValue } from "./client";

export type ReminderType = "MEDICATION" | "CHALLENGE" | "HEALTH_RECORD" | "FAMILY_ALERT" | "SYSTEM";
export type KnownNotificationChannel = "IN_APP" | "EMAIL";
export type NotificationChannel = KnownNotificationChannel | (string & {});
export type NotificationLogStatus = "PENDING" | "SENT" | "FAILED" | "SKIPPED" | "CANCELED";

export type ReminderSchedule = {
  id: number;
  user_id: number;
  reminder_type: ReminderType;
  channel: NotificationChannel;
  title: string;
  message: string;
  related_type: string | null;
  related_id: number | null;
  schedule_time: string | null;
  cron_expression: string | null;
  timezone: string;
  is_active: boolean;
  last_triggered_at: string | null;
  next_trigger_at: string | null;
  created_at: string;
  updated_at: string;
};

export type ReminderSchedulePayload = {
  reminder_type: ReminderType;
  channel?: NotificationChannel;
  title: string;
  message: string;
  related_type?: string | null;
  related_id?: number | null;
  schedule_time?: string | null;
  cron_expression?: string | null;
  timezone?: string;
  is_active?: boolean;
  next_trigger_at?: string | null;
};

export type NotificationLog = {
  id: number;
  user_id: number;
  notification_id: number | null;
  reminder_schedule_id: number | null;
  notification_type: string;
  channel: NotificationChannel;
  title: string;
  message_summary: string | null;
  related_type: string | null;
  related_id: number | null;
  status: NotificationLogStatus;
  provider: string | null;
  provider_message_id: string | null;
  error_code: string | null;
  error_message: string | null;
  sent_at: string | null;
  failed_at: string | null;
  created_at: string;
};

export async function listNotifications<T>(): Promise<T> {
  return apiRequest<T>("/notifications");
}

export async function listUnreadNotifications<T>(): Promise<T> {
  return apiRequest<T>("/notifications/unread");
}

export async function markNotificationRead<T>(notificationId: number): Promise<T> {
  return apiRequest<T>(`/notifications/${notificationId}/read`, { method: "PATCH" });
}

export async function markAllNotificationsRead<T>(): Promise<T> {
  return apiRequest<T>("/notifications/read-all", { method: "PATCH" });
}

export async function listReminderSchedules(params: { isActive?: boolean } = {}): Promise<ReminderSchedule[]> {
  const query = new URLSearchParams();
  query.set("limit", "100");
  if (params.isActive !== undefined) query.set("is_active", String(params.isActive));
  return apiRequest<ReminderSchedule[]>(`/notifications/reminder-schedules?${query.toString()}`);
}

export async function createReminderSchedule(payload: ReminderSchedulePayload): Promise<ReminderSchedule> {
  return apiRequest<ReminderSchedule>("/notifications/reminder-schedules", {
    method: "POST",
    body: payload as Record<string, ApiValue>,
  });
}

export async function updateReminderSchedule(
  scheduleId: number,
  payload: Partial<ReminderSchedulePayload>,
): Promise<ReminderSchedule> {
  return apiRequest<ReminderSchedule>(`/notifications/reminder-schedules/${scheduleId}`, {
    method: "PATCH",
    body: payload as Record<string, ApiValue>,
  });
}

export async function deleteReminderSchedule(scheduleId: number): Promise<{ deleted_count: number }> {
  return apiRequest<{ deleted_count: number }>(`/notifications/reminder-schedules/${scheduleId}`, {
    method: "DELETE",
  });
}

export async function listNotificationLogs(): Promise<NotificationLog[]> {
  return apiRequest<NotificationLog[]>("/notifications/logs?limit=100");
}
