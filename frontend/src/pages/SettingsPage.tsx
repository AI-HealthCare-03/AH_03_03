import { useEffect, useState } from "react";

import { getMySettings, updateMySettings } from "../api/settings";
import Card from "../components/Card";

type Settings = Record<string, unknown>;

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings>({});

  const load = async () => setSettings(await getMySettings<Settings>());

  useEffect(() => {
    void load().catch(() => undefined);
  }, []);

  const toggle = async (key: string) => {
    const next = { [key]: !settings[key] };
    setSettings((prev) => ({ ...prev, ...next }));
    await updateMySettings(next);
    await load();
  };

  return (
    <Card title="설정">
      <div className="settings-list">
        {["notification_enabled", "challenge_reminder_enabled", "medication_reminder_enabled", "diet_reminder_enabled"].map(
          (key) => (
            <label key={key} className="toggle-row">
              <span>{key}</span>
              <input checked={Boolean(settings[key])} onChange={() => void toggle(key)} type="checkbox" />
            </label>
          ),
        )}
        <div className="placeholder">마케팅/민감정보 동의 상세 화면은 후속 구현 예정</div>
      </div>
    </Card>
  );
}
