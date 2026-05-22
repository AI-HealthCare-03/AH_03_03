import { useEffect, useState } from "react";

import { getMySettings, updateMySettings } from "../api/settings";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type Settings = Record<string, unknown>;

const settingLabels: Record<string, string> = {
  notification_enabled: "기본 알림",
  challenge_reminder_enabled: "챌린지 리마인더",
  medication_reminder_enabled: "복약/영양제 리마인더",
  diet_reminder_enabled: "식단 기록 리마인더",
};

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings>({});
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const load = async () => {
    setError("");
    try {
      setSettings(await getMySettings<Settings>());
    } catch (err) {
      setError(err instanceof Error ? err.message : "설정을 불러오지 못했습니다.");
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const toggle = async (key: string) => {
    const next = { [key]: !settings[key] };
    setSettings((prev) => ({ ...prev, ...next }));
    setNotice("");
    try {
      await updateMySettings(next);
      await load();
      setNotice("설정이 저장되었습니다.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "설정 저장에 실패했습니다.");
      setSettings((prev) => ({ ...prev, [key]: settings[key] }));
    }
  };

  return (
    <Card title="설정">
      {error && <ErrorMessage message={error} />}
      {notice && <div className="state-box">{notice}</div>}
      <div className="settings-list">
        {["notification_enabled", "challenge_reminder_enabled", "medication_reminder_enabled", "diet_reminder_enabled"].map(
          (key) => (
            <label key={key} className="toggle-row">
              <span>{settingLabels[key]}</span>
              <input checked={Boolean(settings[key])} onChange={() => void toggle(key)} type="checkbox" />
            </label>
          ),
        )}
        <div className="placeholder">마케팅/민감정보 동의 상세 설정은 준비 중입니다.</div>
      </div>
    </Card>
  );
}
