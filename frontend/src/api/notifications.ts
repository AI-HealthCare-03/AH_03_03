import { apiRequest } from "./client";

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
