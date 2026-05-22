import { FormEvent, useEffect, useMemo, useState } from "react";

import {
  createReminderSchedule,
  deleteReminderSchedule,
  listNotificationLogs,
  listNotifications,
  listReminderSchedules,
  markAllNotificationsRead,
  markNotificationRead,
  updateReminderSchedule,
  type NotificationChannel,
  type NotificationLog,
  type ReminderSchedule,
  type ReminderSchedulePayload,
  type ReminderType,
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
type PageTab = "inbox" | "schedules" | "logs";

const reminderTypeLabels: Record<ReminderType | string, string> = {
  MEDICATION: "복약",
  CHALLENGE: "챌린지",
  HEALTH_RECORD: "건강기록",
  FAMILY_ALERT: "가족 알림",
  SYSTEM: "시스템",
};

const channelLabels: Record<NotificationChannel | string, string> = {
  IN_APP: "앱 알림",
  EMAIL: "이메일",
  SMS: "문자",
  PUSH: "푸시",
  KAKAO: "카카오",
};

const statusLabels: Record<string, string> = {
  PENDING: "대기",
  SENT: "발송 완료",
  FAILED: "실패",
  SKIPPED: "건너뜀",
  CANCELED: "취소",
};

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

const emptyScheduleDraft: ReminderSchedulePayload = {
  reminder_type: "MEDICATION",
  channel: "IN_APP",
  title: "",
  message: "",
  schedule_time: "09:00",
  timezone: "Asia/Seoul",
  is_active: true,
};

function getNotificationType(item: Notification): string {
  return String(item.notification_type ?? item.type ?? "SYSTEM").toUpperCase();
}

export default function NotificationPage() {
  const [activeTab, setActiveTab] = useState<PageTab>("inbox");
  const [items, setItems] = useState<Notification[]>([]);
  const [schedules, setSchedules] = useState<ReminderSchedule[]>([]);
  const [logs, setLogs] = useState<NotificationLog[]>([]);
  const [filter, setFilter] = useState<NotificationFilter>("all");
  const [draft, setDraft] = useState<ReminderSchedulePayload>(emptyScheduleDraft);
  const [editingScheduleId, setEditingScheduleId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [actionLoading, setActionLoading] = useState<number | "all" | "schedule" | null>(null);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const [nextItems, nextSchedules, nextLogs] = await Promise.all([
        listNotifications<Notification[]>(),
        listReminderSchedules(),
        listNotificationLogs(),
      ]);
      setItems(nextItems);
      setSchedules(nextSchedules);
      setLogs(nextLogs);
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

  const submitSchedule = async (event: FormEvent) => {
    event.preventDefault();
    setActionLoading("schedule");
    setError("");
    setNotice("");
    try {
      if (editingScheduleId) {
        await updateReminderSchedule(editingScheduleId, draft);
        setNotice("알림 예약이 수정되었습니다.");
      } else {
        await createReminderSchedule(draft);
        setNotice("알림 예약이 생성되었습니다.");
      }
      setDraft(emptyScheduleDraft);
      setEditingScheduleId(null);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "알림 예약 저장에 실패했습니다.");
    } finally {
      setActionLoading(null);
    }
  };

  const editSchedule = (schedule: ReminderSchedule) => {
    setEditingScheduleId(schedule.id);
    setDraft({
      reminder_type: schedule.reminder_type,
      channel: schedule.channel,
      title: schedule.title,
      message: schedule.message,
      related_type: schedule.related_type,
      related_id: schedule.related_id,
      schedule_time: schedule.schedule_time,
      cron_expression: schedule.cron_expression,
      timezone: schedule.timezone,
      is_active: schedule.is_active,
      next_trigger_at: schedule.next_trigger_at,
    });
  };

  const deactivateSchedule = async (schedule: ReminderSchedule) => {
    setActionLoading(schedule.id);
    setError("");
    setNotice("");
    try {
      await updateReminderSchedule(schedule.id, { is_active: false });
      setNotice("알림 예약이 비활성화되었습니다.");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "알림 예약을 비활성화하지 못했습니다.");
    } finally {
      setActionLoading(null);
    }
  };

  const removeSchedule = async (schedule: ReminderSchedule) => {
    if (!window.confirm("이 알림 예약을 삭제하시겠습니까?")) return;
    setActionLoading(schedule.id);
    setError("");
    setNotice("");
    try {
      await deleteReminderSchedule(schedule.id);
      setNotice("알림 예약이 삭제되었습니다.");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "알림 예약을 삭제하지 못했습니다.");
    } finally {
      setActionLoading(null);
    }
  };

  return (
    <div className="page-stack">
      <Card
        title="알림"
        actions={
          activeTab === "inbox" ? (
            <button disabled={unreadCount === 0 || actionLoading === "all"} onClick={() => void handleMarkAllRead()} type="button">
              모두 읽음
            </button>
          ) : undefined
        }
      >
        <div className="filter-tabs">
          <button className={activeTab === "inbox" ? "filter-tab active" : "filter-tab"} onClick={() => setActiveTab("inbox")} type="button">
            받은 알림
          </button>
          <button className={activeTab === "schedules" ? "filter-tab active" : "filter-tab"} onClick={() => setActiveTab("schedules")} type="button">
            알림 예약
          </button>
          <button className={activeTab === "logs" ? "filter-tab active" : "filter-tab"} onClick={() => setActiveTab("logs")} type="button">
            발송 이력
          </button>
        </div>

        {error && <div className="error-box">{error}</div>}
        {notice && <div className="state-box">{notice}</div>}
        {loading && <div className="state-box">알림 정보를 불러오는 중입니다.</div>}

        {!loading && activeTab === "inbox" && (
          <InboxSection
            actionLoading={actionLoading}
            filter={filter}
            filteredItems={filteredItems}
            onFilterChange={setFilter}
            onMarkRead={handleMarkRead}
            unreadCount={unreadCount}
          />
        )}
        {!loading && activeTab === "schedules" && (
          <ScheduleSection
            actionLoading={actionLoading}
            draft={draft}
            editingScheduleId={editingScheduleId}
            onDeactivate={deactivateSchedule}
            onDraftChange={setDraft}
            onEdit={editSchedule}
            onRemove={removeSchedule}
            onSubmit={submitSchedule}
            onCancelEdit={() => {
              setDraft(emptyScheduleDraft);
              setEditingScheduleId(null);
            }}
            schedules={schedules}
          />
        )}
        {!loading && activeTab === "logs" && <LogSection logs={logs} />}
      </Card>
    </div>
  );
}

function InboxSection({
  actionLoading,
  filter,
  filteredItems,
  onFilterChange,
  onMarkRead,
  unreadCount,
}: {
  actionLoading: number | "all" | "schedule" | null;
  filter: NotificationFilter;
  filteredItems: Notification[];
  onFilterChange: (filter: NotificationFilter) => void;
  onMarkRead: (notificationId: number) => Promise<void>;
  unreadCount: number;
}) {
  return (
    <>
      <div className="filter-tabs sub-tabs">
        <button className={filter === "all" ? "filter-tab active" : "filter-tab"} onClick={() => onFilterChange("all")} type="button">
          전체
        </button>
        <button className={filter === "unread" ? "filter-tab active" : "filter-tab"} onClick={() => onFilterChange("unread")} type="button">
          안 읽음 {unreadCount > 0 ? unreadCount : ""}
        </button>
        <button className={filter === "read" ? "filter-tab active" : "filter-tab"} onClick={() => onFilterChange("read")} type="button">
          읽음
        </button>
      </div>
      {filteredItems.length === 0 ? (
        <div className="empty-state">
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
                  <span className="muted" title={formatDateTime(item.created_at)}>
                    {formatRelativeTime(item.created_at)}
                  </span>
                </div>
                {!item.is_read && (
                  <button
                    className="secondary compact-button"
                    disabled={actionLoading === item.id}
                    onClick={() => void onMarkRead(item.id)}
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
    </>
  );
}

function ScheduleSection({
  actionLoading,
  draft,
  editingScheduleId,
  onCancelEdit,
  onDeactivate,
  onDraftChange,
  onEdit,
  onRemove,
  onSubmit,
  schedules,
}: {
  actionLoading: number | "all" | "schedule" | null;
  draft: ReminderSchedulePayload;
  editingScheduleId: number | null;
  onCancelEdit: () => void;
  onDeactivate: (schedule: ReminderSchedule) => Promise<void>;
  onDraftChange: (draft: ReminderSchedulePayload) => void;
  onEdit: (schedule: ReminderSchedule) => void;
  onRemove: (schedule: ReminderSchedule) => Promise<void>;
  onSubmit: (event: FormEvent) => void;
  schedules: ReminderSchedule[];
}) {
  return (
    <div className="notification-management-grid">
      <section className="mini-card">
        <h3>{editingScheduleId ? "알림 예약 수정" : "새 알림 예약"}</h3>
        <p className="muted">현재는 앱 안에서 확인하는 알림을 중심으로 관리합니다.</p>
        <form className="form" onSubmit={onSubmit}>
          <div className="form-grid">
            <label>
              알림 유형
              <select
                value={draft.reminder_type}
                onChange={(event) => onDraftChange({ ...draft, reminder_type: event.target.value as ReminderType })}
              >
                <option value="MEDICATION">복약</option>
                <option value="CHALLENGE">챌린지</option>
                <option value="HEALTH_RECORD">건강기록</option>
                <option value="SYSTEM">시스템</option>
                <option value="FAMILY_ALERT">가족 알림</option>
              </select>
            </label>
            <label>
              채널
              <select value={draft.channel ?? "IN_APP"} onChange={(event) => onDraftChange({ ...draft, channel: event.target.value as NotificationChannel })}>
                <option value="IN_APP">앱 알림</option>
                <option disabled value="EMAIL">이메일 - 향후 지원</option>
                <option disabled value="SMS">문자 - 향후 지원</option>
                <option disabled value="PUSH">푸시 - 향후 지원</option>
                <option disabled value="KAKAO">카카오 - 향후 지원</option>
              </select>
            </label>
          </div>
          <label>
            제목
            <input value={draft.title} onChange={(event) => onDraftChange({ ...draft, title: event.target.value })} required />
          </label>
          <label>
            메시지
            <textarea value={draft.message} onChange={(event) => onDraftChange({ ...draft, message: event.target.value })} required />
          </label>
          <div className="form-grid">
            <label>
              시간
              <input
                type="time"
                value={(draft.schedule_time ?? "").slice(0, 5)}
                onChange={(event) => onDraftChange({ ...draft, schedule_time: event.target.value || null })}
              />
            </label>
            <label className="toggle-row">
              <span>활성화</span>
              <input
                checked={draft.is_active ?? true}
                onChange={(event) => onDraftChange({ ...draft, is_active: event.target.checked })}
                type="checkbox"
              />
            </label>
          </div>
          <div className="button-row">
            <button disabled={actionLoading === "schedule"} type="submit">
              {editingScheduleId ? "수정 저장" : "예약 생성"}
            </button>
            {editingScheduleId && (
              <button className="secondary" onClick={onCancelEdit} type="button">
                취소
              </button>
            )}
          </div>
        </form>
      </section>

      <section className="notification-list-section">
        {schedules.length === 0 ? (
          <div className="empty-state">
            <strong>등록된 알림 예약이 없습니다.</strong>
            <p>복약, 챌린지, 건강기록 알림을 앱 알림으로 예약해보세요.</p>
          </div>
        ) : (
          <div className="notification-list">
            {schedules.map((schedule) => (
              <article className={schedule.is_active ? "notification-item unread" : "notification-item read"} key={schedule.id}>
                <span className="notification-icon" aria-hidden="true">
                  {notificationTypeIcon[schedule.reminder_type] ?? "🔔"}
                </span>
                <div>
                  <div className="notification-title-row">
                    <strong>{schedule.title}</strong>
                    <span className="badge badge-reference">{reminderTypeLabels[schedule.reminder_type]}</span>
                    <span className="badge badge-reference">{channelLabels[schedule.channel]}</span>
                    <span className={schedule.is_active ? "badge badge-saved" : "badge badge-muted"}>{schedule.is_active ? "활성" : "비활성"}</span>
                  </div>
                  <p>{schedule.message}</p>
                  <span className="muted">
                    {schedule.schedule_time ? `매일 ${schedule.schedule_time.slice(0, 5)}` : "예약 시간 미설정"} · {schedule.timezone}
                  </span>
                </div>
                <div className="notification-action-stack">
                  <button className="secondary compact-button" onClick={() => onEdit(schedule)} type="button">
                    수정
                  </button>
                  {schedule.is_active && (
                    <button
                      className="secondary compact-button"
                      disabled={actionLoading === schedule.id}
                      onClick={() => void onDeactivate(schedule)}
                      type="button"
                    >
                      비활성화
                    </button>
                  )}
                  <button
                    className="btn-danger-outline compact-button"
                    disabled={actionLoading === schedule.id}
                    onClick={() => void onRemove(schedule)}
                    type="button"
                  >
                    삭제
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function LogSection({ logs }: { logs: NotificationLog[] }) {
  if (logs.length === 0) {
    return (
      <div className="empty-state">
        <strong>발송 이력이 없습니다.</strong>
        <p>알림 발송 시도와 처리 결과가 이곳에 기록됩니다.</p>
      </div>
    );
  }

  return (
    <div className="notification-log-list">
      {logs.map((log) => (
        <article className="mini-card" key={log.id}>
          <div className="notification-title-row">
            <strong>{log.title}</strong>
            <span className="badge badge-reference">{channelLabels[log.channel] ?? log.channel}</span>
            <span className={log.status === "SENT" ? "badge badge-saved" : log.status === "FAILED" ? "badge badge-missing" : "badge badge-reference"}>
              {statusLabels[log.status] ?? log.status}
            </span>
          </div>
          <p className="muted">{log.message_summary ?? "요약 메시지가 없습니다."}</p>
          <div className="notification-log-meta">
            <span>유형: {notificationTypeLabel[log.notification_type] ?? reminderTypeLabels[log.notification_type] ?? log.notification_type}</span>
            <span title={formatDateTime(log.created_at)}>생성: {formatRelativeTime(log.created_at)}</span>
            <span title={formatDateTime(log.sent_at)}>발송: {formatRelativeTime(log.sent_at)}</span>
            <span title={formatDateTime(log.failed_at)}>실패: {formatRelativeTime(log.failed_at)}</span>
          </div>
          {log.status === "FAILED" && <div className="state-box">발송 처리 중 오류가 발생했습니다. 잠시 후 다시 확인해주세요.</div>}
        </article>
      ))}
    </div>
  );
}
