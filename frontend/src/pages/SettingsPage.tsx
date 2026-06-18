import { useEffect, useState } from "react";

import { deactivateMe } from "../api/auth";
import { getMySettings, updateMySettings } from "../api/settings";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type Settings = Record<string, unknown>;
type TimeSettingKey = "challenge_reminder_time" | "diet_reminder_time";

const settingLabels: Record<string, string> = {
  notification_enabled: "기본 알림",
  challenge_reminder_enabled: "챌린지 알림",
  medication_reminder_enabled: "복약/영양제 알림",
  diet_reminder_enabled: "식단 기록 알림",
};
const settingDescriptions: Record<string, string> = {
  notification_enabled: "서비스 공지, 건강 분석 완료 등 주요 알림을 받습니다.",
  challenge_reminder_enabled: "참여 중인 챌린지의 오늘 수행 여부 및 완료 알림을 받습니다.",
  medication_reminder_enabled: "등록된 복약/영양제의 복용 시간 알림을 받습니다.",
  diet_reminder_enabled: "식단 기록이 없을 때 기록을 유도하는 알림을 받습니다.",
};

export default function SettingsPage() {
  const { logout } = useAuth();
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

  const handleDeactivate = async () => {
      if (!window.confirm("회원탈퇴를 진행하면 계정이 비활성화됩니다. 계속하시겠습니까?")) return;
      if (!window.confirm("탈퇴 후에는 현재 계정으로 서비스 이용이 제한됩니다. 정말 탈퇴하시겠습니까?")) return;
      try {
        await deactivateMe();
        await logout();
      } catch (err) {
        console.error(err);
      }
    };

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

  const getTimeValue = (key: TimeSettingKey, fallback: string): string => {
    const value = String(settings[key] ?? "").trim();
    if (!value) {
      return fallback;
    }
    const match = /^([01]\d|2[0-3]):([0-5]\d)/.exec(value);
    return match ? `${match[1]}:${match[2]}` : fallback;
  };

  const updateTimeSetting = async (key: TimeSettingKey, value: string) => {
    const nextValue = value || null;
    const previousValue = settings[key];
    setSettings((prev) => ({ ...prev, [key]: nextValue }));
    setNotice("");
    try {
      await updateMySettings({ [key]: nextValue });
      await load();
      setNotice("설정이 저장되었습니다.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "설정 저장에 실패했습니다.");
      setSettings((prev) => ({ ...prev, [key]: previousValue }));
    }
  };

  return (
    <div className="page-stack">
      <header className="dashboard-header">
        <div>
          <h1>설정</h1>
          <p>알림 및 앱 환경설정을 관리합니다.</p>
        </div>
      </header>
    <Card>
      {error && <ErrorMessage message={error} />}
      {notice && <div className="state-box">{notice}</div>}
      <div className="settings-list">
        {["notification_enabled", "challenge_reminder_enabled", "medication_reminder_enabled", "diet_reminder_enabled"].map(
          (key) => (
            <div key={key}>
              <label className="toggle-row">
                <span>
                  <span>{settingLabels[key]}</span>
                  <p className="muted" style={{ fontSize: "13px", marginTop: "2px" }}>{settingDescriptions[key]}</p>
                </span>
                <input checked={Boolean(settings[key])} onChange={() => void toggle(key)} type="checkbox" />
              </label>
              {key === "challenge_reminder_enabled" && Boolean(settings.challenge_reminder_enabled) && (
                <label className="settings-time-row">
                  <span>
                    챌린지 알림 시간
                    <p className="muted" style={{ fontSize: "13px", marginTop: "2px" }}>
                      매일 선택한 시간에 챌린지 리마인더를 보내드려요.
                    </p>
                  </span>
                  <input
                    type="time"
                    value={getTimeValue("challenge_reminder_time", "21:00")}
                    onChange={(event) => void updateTimeSetting("challenge_reminder_time", event.target.value)}
                  />
                </label>
              )}
              {key === "diet_reminder_enabled" && Boolean(settings.diet_reminder_enabled) && (
                <label className="settings-time-row">
                  <span>
                    식단 기록 알림 시간
                    <p className="muted" style={{ fontSize: "13px", marginTop: "2px" }}>
                      선택한 시간에 식단 기록 리마인더를 보내드려요.
                    </p>
                  </span>
                  <input
                    type="time"
                    value={getTimeValue("diet_reminder_time", "20:00")}
                    onChange={(event) => void updateTimeSetting("diet_reminder_time", event.target.value)}
                  />
                </label>
              )}
            </div>
          ),
        )}
        <div className="state-box">
          민감정보와 마케팅 수신 동의는 가입 및 계정 보안 정책에 따라 관리됩니다. 변경이 필요한 경우 1:1 문의로
          요청해주세요.
        </div>
        <div className="state-box">
          알림은 서비스 내부 알림과 이메일 중심으로 제공됩니다.
        </div>
      </div>
      <div style={{ marginTop: "32px", paddingTop: "20px", borderTop: "1px solid var(--color-border)" }}>
        <p className="muted" style={{ marginTop: "4px", marginBottom: "12px" }}>회원탈퇴 시 계정이 비활성화되며 복구가 어렵습니다.</p>
        <button className="danger-ghost" onClick={() => void handleDeactivate()} type="button">
          회원탈퇴
        </button>
      </div>
    </Card>
    </div>
  );
}
