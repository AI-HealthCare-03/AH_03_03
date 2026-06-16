import { useEffect, useMemo, useState } from "react";

import {
  listNotifications,
  markAllNotificationsRead,
  markNotificationRead,
} from "../api/notifications";
import Card from "../components/Card";
import { formatDateTime, formatRelativeTime } from "../utils/format";

type Notification = {
  id: number;
  title?: string;
  message?: string;
  notification_type?: string;
  type?: string;
  is_read?: boolean;
  created_at?: string;
};

type NotificationFilter = "all" | "unread" | "read";

const notificationTypeLabel: Record<string, string> = {
  SYSTEM: "시스템",
  MEDICATION: "복약",
  CHALLENGE: "챌린지",
  HEALTH_RECORD: "건강기록",
  FAMILY_ALERT: "가족",
  ANALYSIS: "분석",
  DIET: "식단",
};

const notificationTypeIcon: Record<string, string> = {
  SYSTEM: "🔔",
  MEDICATION: "💊",
  CHALLENGE: "✅",
  HEALTH_RECORD: "🩺",
  FAMILY_ALERT: "👨‍👩‍👧‍👦",
  ANALYSIS: "📊",
  DIET: "🥗",
};

function getNotificationType(item: Notification): string {
  return String(item.notification_type ?? item.type ?? "SYSTEM").toUpperCase();
}

export default function NotificationPage() {
  const [items, setItems] = useState<Notification[]>([]);
  const [filter, setFilter] = useState<NotificationFilter>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionLoading, setActionLoading] = useState<number | "all" | null>(null);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const nextItems = await listNotifications<Notification[]>();
      setItems(nextItems);
    } catch (err) {
      setError(err instanceof Error ? err.message : "알림 정보를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const filteredItems = useMemo(() => {
    if (filter === "unread") return items.filter((item) => !item.is_read);
    if (filter === "read") return items.filter((item) => item.is_read);
    return items;
  }, [filter, items]);

  const unreadCount = useMemo(() => items.filter((item) => !item.is_read).length, [items]);

  const handleMarkAllRead = async () => {
    setActionLoading("all");
    setError("");
    try {
      await markAllNotificationsRead();
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "알림을 읽음 처리하지 못했습니다.");
    } finally {
      setActionLoading(null);
    }
  };

  const handleMarkRead = async (notificationId: number) => {
    setActionLoading(notificationId);
    setError("");
    try {
      await markNotificationRead(notificationId);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "알림을 읽음 처리하지 못했습니다.");
    } finally {
      setActionLoading(null);
    }
  };

  return (
    <div className="page-stack">
      <Card
        title="알림"
        actions={
          <button disabled={actionLoading === "all"} onClick={() => void handleMarkAllRead()} type="button">
            모두 읽음
          </button>
        }
      >
        {error && <div className="error-box">{error}</div>}

        <div className="filter-tabs">
          <button className={filter === "all" ? "filter-tab active" : "filter-tab"} onClick={() => setFilter("all")} type="button">
            전체
          </button>
          <button className={filter === "unread" ? "filter-tab active" : "filter-tab"} onClick={() => setFilter("unread")} type="button">
            안 읽음 {unreadCount > 0 ? unreadCount : ""}
          </button>
          <button className={filter === "read" ? "filter-tab active" : "filter-tab"} onClick={() => setFilter("read")} type="button">
            읽음
          </button>
        </div>

        {loading && <div className="state-box">알림 정보를 불러오는 중입니다.</div>}

        {!loading && filteredItems.length === 0 ? (
          <div className="empty-state" style={{ marginTop: "16px" }}>
            <strong>표시할 알림이 없습니다.</strong>
            <p>새로운 건강관리 알림이 도착하면 이곳에서 확인할 수 있습니다.</p>
          </div>
        ) : (
          <div className="notification-list">
            {filteredItems.map((item) => {
              const type = getNotificationType(item);
              return (
                <article className={item.is_read ? "notification-item read" : "notification-item unread"} key={String(item.id)}>
                  <span className="notification-icon" aria-hidden="true">
                    {notificationTypeIcon[type] ?? "🔔"}
                  </span>
                  <div>
                    <div className="notification-title-row">
                      <strong>{item.title ?? "알림"}</strong>
                      <span className="badge badge-reference">{notificationTypeLabel[type] ?? "알림"}</span>
                      <span className={item.is_read ? "badge badge-reference" : "badge badge-saved"}>{item.is_read ? "읽음" : "안 읽음"}</span>
                    </div>
                    <p>{item.message ?? "알림 내용을 확인해주세요."}</p>
                    <span className="muted">
                      {formatRelativeTime(item.created_at)} · {formatDateTime(item.created_at)}
                    </span>
                  </div>
                  {!item.is_read && (
                    <button
                      className="secondary compact-button"
                      disabled={actionLoading === item.id}
                      onClick={() => void handleMarkRead(item.id)}
                      type="button"
                    >
                      읽음
                    </button>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </Card>
    </div>
  );
}
