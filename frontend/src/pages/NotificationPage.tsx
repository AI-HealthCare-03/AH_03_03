import { useEffect, useState } from "react";

import { listNotifications, markAllNotificationsRead, markNotificationRead } from "../api/notifications";
import Card from "../components/Card";

type Notification = Record<string, unknown>;

export default function NotificationPage() {
  const [items, setItems] = useState<Notification[]>([]);

  const load = async () => setItems(await listNotifications<Notification[]>());

  useEffect(() => {
    void load().catch(() => undefined);
  }, []);

  return (
    <Card title="알림" actions={<button onClick={() => void markAllNotificationsRead().then(load)}>모두 읽음</button>}>
      <div className="card-list">
        {items.map((item) => (
          <div className="mini-card" key={String(item.id)}>
            <strong>{String(item.title)}</strong>
            <p>{String(item.message)}</p>
            <span>{item.is_read ? "읽음" : "안 읽음"}</span>
            {!item.is_read && <button onClick={() => void markNotificationRead(Number(item.id)).then(load)}>읽음</button>}
          </div>
        ))}
      </div>
    </Card>
  );
}
